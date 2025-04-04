import os
import sys
import pickle
import pandas as pd
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import pyttsx3
import time
import threading
import queue
from typing import List, Dict, Any, Optional, Union
import traceback

# Setup logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("integration/coordinator.log", maxBytes=5 * 1024 * 1024, backupCount=3),  # 5 MB per file, 3 backups
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AgentCoordinator")

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent modules
from health_monitoring.health_monitoring_agent import HealthMonitoringAgent
from saftey_monitoring.safety_monitoring_agent import SafetyMonitoringAgent
from daily_reminder.daily_reminder_agent import DailyReminderAgent
from integration.database import Database
from alerts.alert_system import AlertSystem

class AgentMessage:
    """
    A message class for inter-agent communication
    """
    def __init__(self, 
                sender: str, 
                receiver: str, 
                message_type: str, 
                content: Any,
                priority: int = 1,
                timestamp: Optional[datetime] = None):
        """
        Initialize a new message between agents
        
        Args:
            sender (str): The agent sending the message
            receiver (str): The intended recipient agent
            message_type (str): Type of message (e.g., 'alert', 'request', 'response', 'info')
            content (Any): The message content/payload
            priority (int): Priority level (1-5, with 5 being highest)
            timestamp (datetime, optional): When the message was created, defaults to now
        """
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = timestamp or datetime.now()
        self.id = f"{self.timestamp.strftime('%Y%m%d%H%M%S')}_{sender}_{receiver}_{id(self)}"
        
    def __repr__(self):
        return f"AgentMessage(id={self.id}, sender={self.sender}, receiver={self.receiver}, type={self.message_type}, priority={self.priority})"

class AgentCoordinator:
    """
    Coordinates the activities of all three specialized agents:
    - Health Monitoring Agent
    - Safety Monitoring Agent
    - Daily Reminder Agent
    
    This class serves as the central integration point for the multi-agent system,
    collecting outputs from all agents and determining priority actions.
    """
    
    def __init__(self, database, health_model_dir=None, safety_model_dir=None, reminder_model_dir=None):
        """
        Initialize the AgentCoordinator with a database connection and optional model directories.
        
        Args:
            database: Database connection object
            health_model_dir: Directory for health monitoring models
            safety_model_dir: Directory for safety monitoring models
            reminder_model_dir: Directory for reminder models
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Initialize model directories
        self.health_model_dir = health_model_dir or "models/health"
        self.safety_model_dir = safety_model_dir or "models/safety"
        self.reminder_model_dir = reminder_model_dir or "models/reminder"
        
        # Create model directories if they don't exist
        os.makedirs(self.health_model_dir, exist_ok=True)
        os.makedirs(self.safety_model_dir, exist_ok=True)
        os.makedirs(self.reminder_model_dir, exist_ok=True)
        
        # Initialize agents
        self.health_monitoring_agent = HealthMonitoringAgent(database)
        self.safety_monitoring_agent = SafetyMonitoringAgent(database)
        self.daily_reminder_agent = DailyReminderAgent(database)
        self.alert_system = AlertSystem(database)
        
        # Initialize threads for autonomous operations
        self.autonomous_threads = {}
        self.stop_autonomous_operations = False
        
        # Initialize message queues for inter-agent communication
        self.message_queues = {
            "health_monitoring_agent": queue.PriorityQueue(),
            "safety_monitoring_agent": queue.PriorityQueue(),
            "daily_reminder_agent": queue.PriorityQueue(),
            "coordinator": queue.PriorityQueue()
        }
        
        # Start message processing thread
        self.message_processing_active = True
        self.message_processing_thread = threading.Thread(
            target=self._process_messages,
            daemon=True
        )
        self.message_processing_thread.start()
        
        self.logger.info("AgentCoordinator initialized successfully")
        
        try:
            # Set up severity levels for prioritization
            self.severity_levels = {
                "emergency": 3,
                "urgent": 2,
                "routine": 1,
                "info": 0
            }
            
            self.logger.info("AgentCoordinator initialization complete")
        except Exception as e:
            self.logger.error(f"Error initializing AgentCoordinator: {str(e)}")
            raise
    
    def _process_messages(self):
        """
        Background thread method to process messages from the message queues.
        Continuously monitors all agent message queues and processes messages as they arrive.
        """
        self.logger.info("Starting message processing thread")
        
        while self.message_processing_active:
            # Check all message queues
            for agent_name, agent_queue in self.message_queues.items():
                if not agent_queue.empty():
                    try:
                        # Get message from queue, could be in different formats
                        queue_item = agent_queue.get(False)  # non-blocking
                        
                        # Handle different message formats
                        if isinstance(queue_item, tuple) and len(queue_item) == 3:
                            # Handle format (priority, count, message)
                            message = queue_item[2]
                        elif isinstance(queue_item, dict):
                            # Handle direct dictionary message
                            message = queue_item
                        elif isinstance(queue_item, AgentMessage):
                            # Handle AgentMessage object
                            message = {
                                "sender": queue_item.sender,
                                "receiver": queue_item.receiver,
                                "message_type": queue_item.message_type,
                                "content": queue_item.content,
                                "priority": queue_item.priority,
                                "timestamp": queue_item.timestamp
                            }
                        else:
                            # Unknown format, log and skip
                            self.logger.warning(f"Unknown message format in queue: {type(queue_item)}")
                            agent_queue.task_done()
                            continue
                        
                        if agent_name == "coordinator":
                            # Process messages sent to the coordinator
                            self._handle_coordinator_message(message)
                        else:
                            # Log messages sent to other agents
                            self.logger.info(f"Message for {agent_name}: {message.get('message_type', 'Unknown')} from {message.get('sender', 'Unknown')}")
                            
                            # Implement additional handling for agent-to-agent communication if needed
                            
                        # Mark message as processed
                        agent_queue.task_done()
                    except queue.Empty:
                        # Queue was empty, just continue
                        pass
                    except Exception as e:
                        self.logger.error(f"Error processing message for {agent_name}: {str(e)}")
                        traceback.print_exc()
            
            # Sleep to avoid CPU hogging
            time.sleep(0.1)
        
        self.logger.info("Message processing thread stopped")
    
    def process_all_data(self, user_id, health_data=None, safety_data=None, reminder_data=None, timestamp=None):
        """
        Process data from all agents for a specific user and timestamp.
        
        Args:
            user_id (str): ID of the user to process data for
            health_data (pd.DataFrame): Health monitoring data
            safety_data (pd.DataFrame): Safety monitoring data
            reminder_data (pd.DataFrame): Reminder data
            timestamp (datetime): Timestamp for data processing, defaults to current time
            
        Returns:
            dict: Integrated and prioritized results from all agents
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        self.logger.info(f"Processing data for user {user_id} at {timestamp}")
        
        try:
            # Process data for each agent
            health_results = self._process_health_data(user_id, health_data, timestamp) if health_data is not None else {}
            safety_results = self._process_safety_data(user_id, safety_data, timestamp) if safety_data is not None else {}
            reminder_results = self._process_reminder_data(user_id, reminder_data, timestamp) if reminder_data is not None else {}
            
            # Enable inter-agent communication based on results
            self._coordinate_agent_responses(user_id, health_results, safety_results, reminder_results)
            
            # Combine and prioritize results
            combined_results = self._prioritize_outputs(health_results, safety_results, reminder_results)
            
            # Add metadata
            combined_results["user_id"] = user_id
            combined_results["timestamp"] = timestamp
            
            self.logger.info(f"Completed processing for user {user_id}")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error processing data for user {user_id}: {str(e)}")
            raise
    
    def _process_health_data(self, user_id, health_data, timestamp):
        """
        Process health monitoring data for a user.
        
        Args:
            user_id (str): ID of the user
            health_data (pd.DataFrame): Health monitoring data
            timestamp (datetime): Timestamp for processing
            
        Returns:
            dict: Health monitoring results
        """
        self.logger.info(f"Processing health data for user {user_id}")
        
        try:
            # Get anomaly detection results
            anomalies = self.health_monitoring_agent.detect_anomalies(health_data)
            
            # Get severity predictions
            severities = self.health_monitoring_agent.predict_severity(health_data)
            
            # Combine results
            health_results = {
                "anomalies": anomalies,
                "severities": severities,
                "timestamp": timestamp,
                "user_id": user_id
            }
            
            return health_results
            
        except Exception as e:
            self.logger.error(f"Error processing health data for user {user_id}: {str(e)}")
            return {"error": str(e)}
    
    def _process_safety_data(self, user_id, safety_data, timestamp):
        """
        Process safety monitoring data for a user.
        
        Args:
            user_id (str): ID of the user
            safety_data (pd.DataFrame): Safety monitoring data
            timestamp (datetime): Timestamp for processing
            
        Returns:
            dict: Safety monitoring results
        """
        self.logger.info(f"Processing safety data for user {user_id}")
        
        try:
            # Detect falls or unusual activity
            falls = self.safety_monitoring_agent.detect_falls(safety_data)
            
            # Predict risk levels
            risk_levels = self.safety_monitoring_agent.predict_risk_level(safety_data)
            
            # Combine results
            safety_results = {
                "falls": falls,
                "risk_levels": risk_levels,
                "timestamp": timestamp,
                "user_id": user_id
            }
            
            return safety_results
            
        except Exception as e:
            self.logger.error(f"Error processing safety data for user {user_id}: {str(e)}")
            return {"error": str(e)}
    
    def _process_reminder_data(self, user_id, reminder_data, timestamp):
        """
        Process reminder data for a user.
        
        Args:
            user_id (str): ID of the user
            reminder_data (pd.DataFrame): Reminder data
            timestamp (datetime): Timestamp for processing
            
        Returns:
            dict: Reminder results
        """
        self.logger.info(f"Processing reminder data for user {user_id}")
        
        try:
            # Get priority of reminders
            priorities = self.daily_reminder_agent.predict_priority(reminder_data)
            
            # Get optimal timing for reminders
            optimal_times = self.daily_reminder_agent.predict_optimal_time(reminder_data)
            
            # Predict acknowledgment likelihood
            acknowledgments = self.daily_reminder_agent.predict_acknowledgment(reminder_data)
            
            # Combine results
            reminder_results = {
                "priorities": priorities,
                "optimal_times": optimal_times,
                "acknowledgments": acknowledgments,
                "timestamp": timestamp,
                "user_id": user_id
            }
            
            return reminder_results
            
        except Exception as e:
            self.logger.error(f"Error processing reminder data for user {user_id}: {str(e)}")
            return {"error": str(e)}
    
    def _coordinate_agent_responses(self, user_id, health_results, safety_results, reminder_results):
        """
        Coordinate responses between agents based on their individual findings
        
        Args:
            user_id (str): The user ID
            health_results (dict): Results from health monitoring
            safety_results (dict): Results from safety monitoring
            reminder_results (dict): Results from daily reminder
        """
        try:
            alerts = []
            
            # Check if there are health anomalies
            if health_results and health_results.get("anomalies"):
                for i, anomaly in enumerate(health_results.get("anomalies", [])):
                    if anomaly:
                        metric = list(health_results.get("metrics", {}).keys())[i]
                        alerts.append(f"Health anomaly: {metric}")
                        
                        # Send message to safety agent to correlate with safety data
                        self.send_message(
                            sender="health_monitoring_agent",
                            receiver="coordinator",
                            message_type="health_alert",
                            content={
                                "user_id": user_id,
                                "health_issue": metric,
                                "severity": health_results.get("severities", [])[i] if i < len(health_results.get("severities", [])) else "unknown",
                                "timestamp": datetime.now()
                            },
                            priority=4 if metric in ["heart_rate", "blood_pressure", "oxygen_saturation"] else 3
                        )
            
            # Check if there are safety concerns
            if safety_results and safety_results.get("falls"):
                alerts.append("Fall detected")
                
                # Send high-priority message about fall detection
                self.send_message(
                    sender="safety_monitoring_agent",
                    receiver="coordinator",
                    message_type="safety_alert",
                    content={
                        "user_id": user_id,
                        "event_type": "fall_detected",
                        "timestamp": datetime.now(),
                        "location": safety_results.get("location", "unknown")
                    },
                    priority=5  # Highest priority
                )
            
            # Check for missed reminders
            if reminder_results and "missed" in reminder_results:
                for reminder in reminder_results.get("missed", []):
                    alerts.append(f"Missed reminder: {reminder.get('title')}")
                    
                    # Send notification about missed reminder
                    self.send_message(
                        sender="daily_reminder_agent",
                        receiver="coordinator",
                        message_type="reminder_status",
                        content={
                            "user_id": user_id,
                            "reminder_id": reminder.get("id"),
                            "title": reminder.get("title"),
                            "reminder_type": reminder.get("reminder_type"),
                            "scheduled_time": reminder.get("scheduled_time"),
                            "status": "missed"
                        },
                        priority=3
                    )
            
            # If multiple issues detected, create a combined alert
            if len(alerts) > 1:
                self.send_message(
                    sender="coordinator",
                    receiver="coordinator",
                    message_type="combined_alert",
                    content={
                        "user_id": user_id,
                        "alerts": alerts
                    },
                    priority=4  # High priority for combined issues
                )
                
        except Exception as e:
            self.logger.error(f"Error coordinating agent responses: {str(e)}")
    
    def _prioritize_outputs(self, health_results, safety_results, reminder_results):
        """
        Combine and prioritize results from all agents based on severity and urgency.
        """
        self.logger.info("Prioritizing outputs from all agents")
        
        # Initialize list to hold all alerts/actions
        all_actions = []
        
        # Process health results
        if health_results and "anomalies" in health_results:
            for i, anomaly in enumerate(health_results.get("anomalies", [])):
                if anomaly:
                    severity = "urgent"
                    if "severities" in health_results and i < len(health_results["severities"]):
                        severity = "emergency" if health_results["severities"][i] > 0.7 else "urgent"
                    
                    action = {
                        "type": "health_alert",
                        "severity": severity,
                        "severity_level": self.severity_levels[severity],
                        "message": f"Health anomaly detected with severity {health_results['severities'][i] if 'severities' in health_results and i < len(health_results['severities']) else 'unknown'}",
                        "data": {"anomaly_index": i, "severity": health_results.get("severities", [0])[i] if i < len(health_results.get("severities", [])) else 0}
                    }
                    all_actions.append(action)
                    
                    # Log high-priority actions
                    if severity in ["emergency", "urgent"]:
                        self.logger.info(f"High-priority action: {action['message']}")
        
        # Process safety results
        if safety_results and "falls" in safety_results:
            for i, fall in enumerate(safety_results.get("falls", [])):
                if fall:
                    severity = "emergency"
                    action = {
                        "type": "safety_alert",
                        "severity": severity,
                        "severity_level": self.severity_levels[severity],
                        "message": "Fall detected",
                        "data": {"fall_index": i}
                    }
                    all_actions.append(action)
                    
                    # Log high-priority actions
                    if severity in ["emergency", "urgent"]:
                        self.logger.info(f"High-priority action: {action['message']}")
        
        if safety_results and "risk_levels" in safety_results:
            for i, risk in enumerate(safety_results.get("risk_levels", [])):
                if risk > 0.5:
                    severity = "urgent" if risk > 0.7 else "routine"
                    action = {
                        "type": "safety_warning",
                        "severity": severity,
                        "severity_level": self.severity_levels[severity],
                        "message": f"Safety risk detected with level {risk}",
                        "data": {"risk_index": i, "risk_level": risk}
                    }
                    all_actions.append(action)
                    
                    # Log high-priority actions
                    if severity in ["emergency", "urgent"]:
                        self.logger.info(f"High-priority action: {action['message']}")
        
        # Process reminder results
        if reminder_results and "priorities" in reminder_results:
            for i, priority in enumerate(reminder_results.get("priorities", [])):
                severity = "routine"
                if priority > 0.7:
                    severity = "urgent"
                
                action = {
                    "type": "reminder",
                    "severity": severity,
                    "severity_level": self.severity_levels[severity],
                    "message": f"Reminder with priority {priority}",
                    "data": {
                        "priority": priority,
                        "optimal_time": reminder_results.get("optimal_times", [None])[i] if i < len(reminder_results.get("optimal_times", [])) else None,
                        "acknowledgment_likelihood": reminder_results.get("acknowledgments", [None])[i] if i < len(reminder_results.get("acknowledgments", [])) else None
                    }
                }
                all_actions.append(action)
                
                # Log high-priority actions
                if severity in ["emergency", "urgent"]:
                    self.logger.info(f"High-priority action: {action['message']}")
        
        # Sort actions by severity level (descending)
        all_actions.sort(key=lambda x: x["severity_level"], reverse=True)
        
        # Compile final output
        prioritized_output = {
            "actions": all_actions,
            "high_priority_count": sum(1 for action in all_actions if action["severity"] in ["emergency", "urgent"]),
            "requires_immediate_attention": any(action["severity"] == "emergency" for action in all_actions)
        }
        
        logger.info(f"Prioritization complete. {len(all_actions)} actions identified.")
        return prioritized_output
    
    def get_agent_status(self):
        """
        Get the status of all agents.
        
        Returns:
            dict: Status information for all agents
        """
        return {
            "health_agent": {
                "status": "active" if hasattr(self, "health_monitoring_agent") else "inactive",
                "models_loaded": self.health_monitoring_agent.models_loaded if hasattr(self, "health_monitoring_agent") else False
            },
            "safety_agent": {
                "status": "active" if hasattr(self, "safety_monitoring_agent") else "inactive",
                "models_loaded": self.safety_monitoring_agent.models_loaded if hasattr(self, "safety_monitoring_agent") else False
            },
            "reminder_agent": {
                "status": "active" if hasattr(self, "daily_reminder_agent") else "inactive",
                "models_loaded": self.daily_reminder_agent.models_loaded if hasattr(self, "daily_reminder_agent") else False
            }
        }
    
    def generate_recommendations(self, user_id, health_results, safety_results, reminder_results):
        """
        Generate personalized recommendations based on data from all agents.
        
        Args:
            user_id (str): ID of the user
            health_results (dict): Results from health monitoring
            safety_results (dict): Results from safety monitoring
            reminder_results (dict): Results from reminder agent
            
        Returns:
            list: Personalized recommendations
        """
        logger.info(f"Generating recommendations for user {user_id}")
        
        recommendations = []
        
        # Health-based recommendations
        if health_results and "anomalies" in health_results and any(health_results["anomalies"]):
            recommendations.append({
                "type": "health",
                "message": "Consider scheduling a check-up with your healthcare provider.",
                "priority": "high" if any(sev > 0.7 for sev in health_results.get("severities", [])) else "medium"
            })
        
        # Safety-based recommendations
        if safety_results and "risk_levels" in safety_results and any(level > 0.5 for level in safety_results["risk_levels"]):
            recommendations.append({
                "type": "safety",
                "message": "Consider reviewing your home environment for potential hazards.",
                "priority": "high" if any(level > 0.7 for level in safety_results["risk_levels"]) else "medium"
            })
        
        # Reminder-based recommendations
        if reminder_results and "acknowledgments" in reminder_results:
            low_ack = [i for i, ack in enumerate(reminder_results["acknowledgments"]) if ack < 0.5]
            if low_ack:
                recommendations.append({
                    "type": "reminder",
                    "message": "Consider adjusting reminder timings to improve acknowledgment rates.",
                    "priority": "medium"
                })
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def summarize_user_status(self, user_id, history_period=7):
        """
        Summarize user status based on recent historical data.
        
        Args:
            user_id (str): ID of the user
            history_period (int): Number of days of history to include
            
        Returns:
            dict: Summary of user status
        """
        logger.info(f"Generating status summary for user {user_id} over past {history_period} days")
        
        # This would typically fetch historical data from a database
        # For now, we'll return a placeholder
        
        return {
            "user_id": user_id,
            "period": f"Last {history_period} days",
            "health_trends": {
                "anomalies_detected": 0,
                "severity_trend": "stable"
            },
            "safety_trends": {
                "falls_detected": 0,
                "risk_level_trend": "stable"
            },
            "reminder_trends": {
                "reminders_issued": 0,
                "acknowledgment_rate": 0.0
            }
        }
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get health anomalies for all users"""
        return self.health_monitoring_agent.get_anomalies()

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get safety alerts for all users"""
        return self.safety_monitoring_agent.get_alerts()

    def get_reminders(self) -> List[Dict[str, Any]]:
        """Get all reminders"""
        return self.daily_reminder_agent.get_reminders()

    def get_upcoming_reminders(self) -> List[Dict[str, Any]]:
        """Get upcoming reminders"""
        return self.daily_reminder_agent.get_upcoming_reminders()

    def get_health_overview(self) -> Dict[str, Any]:
        """Get health overview for all users"""
        return self.health_monitoring_agent.get_health_overview()

    def get_health_report(self, user_id: int) -> Dict[str, Any]:
        """Get health report for a specific user"""
        return self.health_monitoring_agent.get_health_report(user_id)

    def get_users_with_abnormal_readings(self) -> List[Dict[str, Any]]:
        """Get users with abnormal health readings"""
        return self.health_monitoring_agent.get_users_with_abnormal_readings()

    def schedule_periodic_checks(self):
        """Schedule periodic autonomous checks for all agents"""
        import threading
        import time
        
        # Health monitoring thread
        def health_monitoring_thread():
            while not self.stop_autonomous_operations:
                try:
                    logger.info("Running periodic health monitoring check")
                    if self.health_monitoring_agent:
                        # Run health check with anomaly detection
                        results = self.health_monitoring_agent.run_health_check(
                            use_ml_models=True,  # Enable ML-based anomaly detection
                            predict_severity=True  # Use ML to predict severity
                        )
                        
                        # Process any detected anomalies
                        if results and 'anomalies' in results and results['anomalies']:
                            for anomaly in results['anomalies']:
                                # Send message to coordinator about health anomaly
                                self.send_message(AgentMessage(
                                    sender="health_monitoring_agent",
                                    receiver="coordinator",
                                    message_type="HEALTH_ALERT",
                                    content=anomaly,
                                    priority=3 if anomaly.get('severity') == 'high' else 2
                                ))
                                
                except Exception as e:
                    logger.error(f"Error in health monitoring thread: {str(e)}")
                
                # Sleep for 30 minutes before next check
                for _ in range(30 * 60 // 10):  # Check for stop signal every 10 seconds
                    if self.stop_autonomous_operations:
                        break
                    time.sleep(10)
                    
        # Safety monitoring thread
        def safety_monitoring_thread():
            while not self.stop_autonomous_operations:
                try:
                    logger.info("Running periodic safety monitoring check")
                    if self.safety_monitoring_agent:
                        # Run safety check with fall detection and risk assessment
                        results = self.safety_monitoring_agent.monitor_safety(
                            use_ml_detection=True,  # Enable ML-based fall detection
                            predict_risk=True  # Use ML to assess risk levels
                        )
                        
                        # Process any detected issues
                        if results and 'issues' in results and results['issues']:
                            for issue in results['issues']:
                                # Send message to coordinator about safety issue
                                self.send_message(AgentMessage(
                                    sender="safety_monitoring_agent",
                                    receiver="coordinator",
                                    message_type="SAFETY_ALERT",
                                    content=issue,
                                    priority=3 if issue.get('severity') == 'high' else 2
                                ))
                                
                except Exception as e:
                    logger.error(f"Error in safety monitoring thread: {str(e)}")
                
                # Sleep for 10 minutes before next check
                for _ in range(10 * 60 // 10):  # Check for stop signal every 10 seconds
                    if self.stop_autonomous_operations:
                        break
                    time.sleep(10)
                    
        # Reminder thread
        def reminder_thread():
            while not self.stop_autonomous_operations:
                try:
                    self.logger.info("Running periodic reminder checks for all users...")
                    
                    # Get all users directly from the database connection
                    cursor = self.database.cursor()
                    cursor.execute("SELECT * FROM users")
                    users = cursor.fetchall()
                    current_time = datetime.now()
                    
                    for user in users:
                        # Get user ID from the result (assuming id is the first column)
                        user_id = user[0]
                        
                        # Get reminders that are due in the next hour but haven't been acknowledged
                        one_hour_later = (current_time + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
                        
                        try:
                            # Get pending reminders for the next hour directly from the database
                            cursor = self.database.cursor()
                            cursor.execute("""
                                SELECT * FROM reminders 
                                WHERE user_id = ? 
                                AND scheduled_time BETWEEN ? AND ?
                                AND (is_acknowledged IS NULL OR is_acknowledged = 0)
                            """, (user_id, current_time.strftime('%Y-%m-%d %H:%M:%S'), one_hour_later))
                            
                            reminders = cursor.fetchall()
                            
                            if reminders:
                                for reminder in reminders:
                                    # Convert reminder to a dictionary
                                    reminder_dict = dict(zip([column[0] for column in cursor.description], reminder))
                                    
                                    # Reminder is due within 15 minutes
                                    reminder_time = datetime.strptime(reminder_dict['scheduled_time'], '%Y-%m-%d %H:%M:%S')
                                    mins_until_due = (reminder_time - current_time).total_seconds() / 60
                                    
                                    if 0 <= mins_until_due <= 15:
                                        # Create an alert for the upcoming reminder
                                        self.send_message(
                                            sender="daily_reminder_agent",
                                            receiver="coordinator",
                                            message_type="REMINDER_STATUS",
                                            content={
                                                "user_id": user_id,
                                                "reminder_id": reminder_dict['id'],
                                                "title": reminder_dict['title'],
                                                "reminder_type": reminder_dict.get('reminder_type', 'general'),
                                                "scheduled_time": reminder_dict['scheduled_time'],
                                                "status": "pending"
                                            },
                                            priority=3 if reminder_dict.get('priority', 1) == 3 else 2
                                        )
                                        self.logger.info(f"Created reminder status alert for user {user_id}")
                        
                        except Exception as e:
                            self.logger.error(f"Error processing reminders for user {user_id}: {str(e)}")
                    
                    # Check for missed reminders (past due and not acknowledged)
                    try:
                        # Get all users' missed reminders from the past 24 hours
                        yesterday = (current_time - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                        
                        for user in users:
                            user_id = user[0]
                            
                            # Get missed reminders directly from the database
                            cursor = self.database.cursor()
                            cursor.execute("""
                                SELECT * FROM reminders 
                                WHERE user_id = ? 
                                AND scheduled_time BETWEEN ? AND ?
                                AND (is_acknowledged IS NULL OR is_acknowledged = 0)
                            """, (user_id, yesterday, current_time.strftime('%Y-%m-%d %H:%M:%S')))
                            
                            missed_reminders = cursor.fetchall()
                            
                            if missed_reminders:
                                # Convert reminders to dictionaries
                                missed_reminders = [dict(zip([column[0] for column in cursor.description], reminder)) 
                                                 for reminder in missed_reminders]
                                
                                # Group reminders by type for better notifications
                                reminders_by_type = {}
                                for reminder in missed_reminders:
                                    reminder_type = reminder.get('reminder_type', 'other')
                                    if reminder_type not in reminders_by_type:
                                        reminders_by_type[reminder_type] = []
                                    reminders_by_type[reminder_type].append(reminder)
                                
                                # Create alerts for missed reminders
                                for reminder_type, reminders in reminders_by_type.items():
                                    if reminder_type == 'medication':
                                        severity = "high"
                                    elif reminder_type in ['appointment', 'exercise']:
                                        severity = "medium"
                                    else:
                                        severity = "info"
                                        
                                    reminder_titles = [r['title'] for r in reminders]
                                    self.send_message(
                                        sender="daily_reminder_agent",
                                        receiver="coordinator",
                                        message_type="REMINDER_STATUS",
                                        content={
                                            "user_id": user_id,
                                            "reminder_id": [r['id'] for r in reminders],
                                            "title": reminder_titles,
                                            "reminder_type": [r.get('reminder_type', 'general') for r in reminders],
                                            "scheduled_time": [r['scheduled_time'] for r in reminders],
                                            "status": "missed"
                                        },
                                        priority=3 if severity == "high" else 2
                                    )
                                    self.logger.info(f"Created missed reminder status alerts for {len(reminders)} {reminder_type} reminders for user {user_id}")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing missed reminders: {str(e)}")
                    
                    self.logger.info("Completed scheduled reminder checks for all users")
                    
                except Exception as e:
                    self.logger.error(f"Error in reminder thread: {str(e)}")
                
                # Sleep for 5 minutes before next check
                for _ in range(5 * 60 // 10):  # Check for stop signal every 10 seconds
                    if self.stop_autonomous_operations:
                        break
                    time.sleep(10)
                    
        # Schedule threads
        self.autonomous_threads['health_monitoring'] = threading.Thread(target=health_monitoring_thread)
        self.autonomous_threads['safety_monitoring'] = threading.Thread(target=safety_monitoring_thread)
        self.autonomous_threads['reminder_checks'] = threading.Thread(target=reminder_thread)
        
        # Start all threads
        for thread in self.autonomous_threads.values():
            thread.start()
        
        self.logger.info("All autonomous agent threads are running")
        return True
    
    def _handle_coordinator_message(self, message):
        """
        Handle messages sent to the coordinator.
        
        Args:
            message (dict): Message data containing message_type and content
        """
        try:
            message_type = message.get("message_type")
            content = message.get("content", {})
            sender = message.get("sender")
            user_id = content.get("user_id")
            
            self.logger.info(f"Handling coordinator message: {message_type} from {sender}")
            
            if message_type == "health_alert":
                # Process health alert
                self.logger.info(f"Health alert received for user {user_id}")
                severity = content.get("severity", "low")
                
                # Create alert in the system
                alert_data = {
                    "user_id": user_id,
                    "type": "health",
                    "message": content.get("message", "Health anomaly detected"),
                    "severity": severity,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "acknowledged": False,
                    "source": "health_monitoring_agent"
                }
                
                # Save alert to database
                alert_id = self.alert_system.create_alert(alert_data)
                
                # Send voice alert for high priority
                if severity == "high":
                    self._send_voice_alert("health", alert_data, user_id)
                
            elif message_type == "safety_alert":
                # Process safety alert
                self.logger.info(f"Safety alert received for user {user_id}")
                severity = content.get("severity", "low")
                
                # Create alert in the system
                alert_data = {
                    "user_id": user_id,
                    "type": "safety",
                    "message": content.get("message", "Safety issue detected"),
                    "severity": severity,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "acknowledged": False,
                    "source": "safety_monitoring_agent"
                }
                
                # Save alert to database
                alert_id = self.alert_system.create_alert(alert_data)
                
                # Send voice alert for high priority
                if severity == "high":
                    self._send_voice_alert("safety", alert_data, user_id)
                
            elif message_type == "reminder_status":
                # Process reminder status update
                self.logger.info(f"Reminder status update for user {user_id}")
                
                # Handle reminder status update
                reminder_id = content.get("reminder_id")
                status = content.get("status")
                
                if reminder_id and status:
                    # Update reminder status in database if needed
                    pass
                
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling coordinator message: {str(e)}")
            traceback.print_exc()

    def _send_voice_alert(self, message_type, alert_data, user_id=None):
        """
        Send an immediate voice alert for high priority issues.
        
        Args:
            message_type (str): Type of message ('health', 'safety', etc.)
            alert_data (dict): Alert information
            user_id (int, optional): User ID
        """
        try:
            # Import the voice reminder service
            from backend.voice_reminder_service import get_voice_reminder_service
            
            # Get user name if available
            user_name = "User"
            if self.database and user_id:
                try:
                    cursor = self.database.cursor()
                    cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
                    result = cursor.fetchone()
                    if result and 'name' in result:
                        user_name = result['name']
                except Exception as e:
                    logger.error(f"Error getting user name: {e}")
            
            # Construct alert message
            if message_type == "health":
                alert_type = alert_data.get('alert_type', 'Health issue')
                description = alert_data.get('description', 'A health issue has been detected')
                
                alert_message = (
                    f"URGENT HEALTH ALERT for {user_name}. {alert_type}. {description}. "
                    f"Please take immediate action and check the system for details."
                )
            elif message_type == "safety":
                alert_type = alert_data.get('alert_type', 'Safety issue')
                description = alert_data.get('description', 'A safety issue has been detected')
                
                alert_message = (
                    f"URGENT SAFETY ALERT for {user_name}. {alert_type}. {description}. "
                    f"Please check on {user_name} immediately."
                )
            elif message_type == "combined":
                description = alert_data.get('description', 'Multiple issues detected')
                
                alert_message = (
                    f"CRITICAL ALERT for {user_name}. {description}. "
                    f"Multiple systems have detected issues. Immediate attention required."
                )
            else:
                alert_message = f"URGENT ALERT for {user_name}. Please check the system immediately."
            
            # Get voice service
            voice_service = get_voice_reminder_service()
            
            # Queue the alert with high priority settings
            settings = {
                'rate': 130,  # Slower for urgent alerts
                'volume': 1.0,  # Full volume
                'play_audio': True,
                'save_audio': True,
                # Repeat the message 2 times for urgency
                'repeat': 2
            }
            
            # Send the alert
            success = voice_service.queue_reminder(
                text=alert_message,
                user_id=user_id,
                reminder_id=None,
                settings=settings
            )
            
            if success:
                logger.info(f"Voice alert sent: {alert_message}")
            else:
                logger.warning(f"Failed to send voice alert")
                
            return success
            
        except ImportError:
            # Fall back to direct reminder agent if service not available
            if hasattr(self, 'daily_reminder_agent') and self.daily_reminder_agent:
                try:
                    alert_message = f"URGENT ALERT: {message_type.upper()} ISSUE DETECTED. Please check the system immediately."
                    self.daily_reminder_agent.send_voice_reminder(alert_message)
                    logger.info(f"Voice alert sent using fallback method")
                    return True
                except Exception as e:
                    logger.error(f"Error sending voice alert using fallback method: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending voice alert: {e}")
            return False

    def _coordinate_agent_responses(self, message):
        """Coordinate responses based on messages from multiple agents"""
        # Track alerts by user ID
        user_id = message.content.get('user_id', 1)
        
        # Add message to alert tracking
        if not hasattr(self, '_recent_alerts'):
            self._recent_alerts = {}
            
        if user_id not in self._recent_alerts:
            self._recent_alerts[user_id] = []
            
        self._recent_alerts[user_id].append({
            'type': message.message_type,
            'content': message.content,
            'timestamp': datetime.now(),
            'from_agent': message.sender
        })
        
        # Clean up old alerts (older than 30 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=30)
        self._recent_alerts[user_id] = [
            alert for alert in self._recent_alerts[user_id] 
            if alert['timestamp'] > cutoff_time
        ]
        
        # Check for concerning patterns requiring coordinated response
        recent_alerts = self._recent_alerts[user_id]
        
        # If we have multiple alerts from different agents for the same user
        if len(recent_alerts) >= 2:
            agent_sources = set(alert['from_agent'] for alert in recent_alerts)
            
            # If alerts coming from multiple agents
            if len(agent_sources) >= 2:
                # Create a combined high-priority alert
                combined_alert = {
                    'user_id': user_id,
                    'alert_type': 'multiple_issues_detected',
                    'description': f"Multiple issues detected by {', '.join(agent_sources)}",
                    'severity': 'high',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source_alerts': recent_alerts
                }
                
                # Log the coordinated response
                logger.warning(f"Coordinated response triggered for user {user_id}: {combined_alert['description']}")
                
                # Send voice alert for the combined critical situation
                self._send_voice_alert(
                    message_type="combined",
                    alert_data=combined_alert,
                    user_id=user_id
                )
                
                # Consider this as a high-priority situation
                # 1. Create immediate high-priority reminder
                if self.daily_reminder_agent:
                    try:
                        # Directly create a reminder in the database
                        if self.daily_reminder_agent.database:
                            cursor = self.daily_reminder_agent.database.cursor()
                            cursor.execute("""
                                INSERT INTO reminders
                                (user_id, title, description, reminder_type, scheduled_time, priority, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                user_id, 
                                "URGENT: Multiple issues detected",
                                combined_alert['description'],
                                "alert",
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "high",
                                "pending"
                            ))
                            self.daily_reminder_agent.database.commit()
                            
                            reminder_id = cursor.lastrowid
                            logger.info(f"Created high-priority reminder ID {reminder_id} for multiple issues")
                    except Exception as e:
                        logger.error(f"Error creating coordinated response reminder: {str(e)}")
                        
                # 2. Intensify monitoring from all agents
                for agent in ["health_monitoring_agent", "safety_monitoring_agent"]:
                    self.send_message(
                        sender="coordinator",
                        receiver=agent,
                        message_type="INCREASE_MONITORING",
                        content={
                            "reason": "multiple_issues_detected",
                            "duration_minutes": 60,
                            "alert_data": combined_alert
                        },
                        priority=3  # Highest priority
                    )
                    
                # Reset the alerts as we've processed them into a combined response
                self._recent_alerts[user_id] = []

    def send_message(self, sender=None, receiver=None, message_type=None, content=None, priority=1, message=None):
        """
        Send a message from one agent to another or to the coordinator.
        
        Args:
            sender (str): The agent sending the message
            receiver (str): The intended recipient agent
            message_type (str): Type of message (e.g., 'alert', 'request', 'response', 'info')
            content (Any): The message content/payload
            priority (int): Priority level (1-5, with 5 being highest)
            message (AgentMessage, optional): An AgentMessage object to send directly
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        try:
            # Handle the case where an AgentMessage object is passed directly
            if isinstance(sender, AgentMessage):
                message = sender
                sender = message.sender
                receiver = message.receiver
                message_type = message.message_type
                content = message.content
                priority = message.priority
                
            # Validate receiver exists
            if receiver not in self.message_queues:
                self.logger.error(f"Cannot send message to unknown receiver: {receiver}")
                return False
                
            # Create message if not provided directly
            if message is None:
                message = {
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": message_type,
                    "content": content,
                    "priority": priority,
                    "timestamp": datetime.now()
                }
            
            # Log the message
            self.logger.info(f"Sending message: {message_type} from {sender} to {receiver} with priority {priority}")
            
            # Add to recipient's queue with priority-based ordering (using negative priority so higher values come first)
            self.message_queues[receiver].put((-priority, time.time(), message))
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    logger.info("Starting Agent Coordinator")

    try:
        # Initialize the AgentCoordinator
        coordinator = AgentCoordinator()
        logger.info("Agent Coordinator initialized successfully")

        # Production-ready: Replace with API calls or scheduled tasks
        logger.info("Agent Coordinator is running. Waiting for tasks...")
        while True:
            # Example: Perform periodic maintenance or data processing
            time.sleep(3600)  # Sleep for 1 hour

    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        logger.info("Agent Coordinator stopped")