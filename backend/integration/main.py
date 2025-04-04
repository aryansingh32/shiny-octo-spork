import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import importlib.util
import json
import traceback
import time
import sqlite3
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from integration.agent_coordinator import AgentCoordinator
from integration.alert_system import AlertSystem
from integration.database import Database
from typing import List, Dict

# Load environment variables from the correct .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend.env')
load_dotenv(env_path)

# Setup logging with rotation
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, "elderly_care_system.log"),
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ElderlyCareSys")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Flask Configuration
app.config["ENV"] = os.getenv("FLASK_ENV", "production")
app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "False").lower() == "true"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "default_secret_key")

class ElderlyCareSys:
    """Main application class for the Elderly Care Multi-Agent System."""
    
    def __init__(self, config_path=None, db_path=None, start_autonomous=True):
        """Initialize the Elderly Care System with all its components."""
        self.logger = logging.getLogger("ElderlyCareSys.Main")
        self.logger.info("Initializing Elderly Care System")
        
        try:
            # Load configuration if provided
            self.config = self._load_config(config_path)
            
            # Initialize database with proper error handling
            db_path = db_path or os.getenv("DATABASE_PATH", "elderly_care.db")
            self.db = Database(db_path=db_path)
            self.database = self.db
            self.logger.info("Database initialized successfully")
        
            # Initialize alert system with the database connection
            self.alert_system = AlertSystem(db_connection=self.db.conn)
            self.logger.info("Alert system initialized successfully")
        
            # Initialize agent coordinator
            self.agent_coordinator = AgentCoordinator(
                database=self.db.conn,
                health_model_dir=self.config.get('model_paths', {}).get('health', 'health_monitoring/models/'),
                safety_model_dir=self.config.get('model_paths', {}).get('safety', 'saftey_monitoring/modelsSafety/'),
                reminder_model_dir=self.config.get('model_paths', {}).get('reminder', 'daily_reminder/models/')
            )
            self.logger.info("Agent coordinator initialized successfully")
            
            # Start autonomous agent operations if requested
            if start_autonomous:
                self.start_autonomous_operations()
            
            self.logger.info("Elderly Care System initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Elderly Care System: {e}")
            if hasattr(self, 'db'):
                self.db.close()
            raise
    
    def _load_config(self, config_path):
        """Load configuration from a JSON file or environment variables."""
        load_dotenv(env_path)

        default_config = {
            "agent_paths": {
                "health": os.getenv("HEALTH_AGENT_PATH", "health_monitoring/health_monetoring_agent.py"),
                "safety": os.getenv("SAFETY_AGENT_PATH", "saftey_monitoring/saftey_monetoring_agent.py"),
                "reminder": os.getenv("REMINDER_AGENT_PATH", "daily_reminder/daily_remainder_agent.py")
            },
            "alert_settings": {
                "emergency_cooldown_minutes": int(os.getenv("EMERGENCY_COOLDOWN", 5)),
                "urgent_cooldown_minutes": int(os.getenv("URGENT_COOLDOWN", 30)),
                "routine_cooldown_minutes": int(os.getenv("ROUTINE_COOLDOWN", 120))
            },
            "monitoring_intervals": {
                "health_minutes": int(os.getenv("HEALTH_INTERVAL", 60)),
                "safety_minutes": int(os.getenv("SAFETY_INTERVAL", 15)),
                "reminder_minutes": int(os.getenv("REMINDER_INTERVAL", 5))
            },
            "data_retention_days": int(os.getenv("DATA_RETENTION_DAYS", 90))
        }

        if not config_path:
            self.logger.info("No configuration file provided, using defaults")
            return default_config

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Configuration loaded from {config_path}")

                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in config[key]:
                                config[key][sub_key] = sub_value

                return config
        except Exception as e:
            self.logger.warning(f"Failed to load configuration from {config_path}: {e}")
            self.logger.info("Using default configuration")
            return default_config
    
    def process_data_for_user(self, user_id):
        """
        Process data for a specific user.

        Args:
            user_id (int): The ID of the user to process data for.

        Returns:
            dict: A dictionary containing the results of the processing.
        """
        self.logger.info(f"Processing data for user {user_id}")
        try:
            # Fetch user details from the database
            user = self.db.get_user(user_id)
            if not user:
                self.logger.error(f"User with ID {user_id} not found")
                return {"error": f"User with ID {user_id} not found"}

            # Example: Process user data (this can be extended as needed)
            self.logger.info(f"User details: {user}")
            # Perform additional processing here...

            return {"success": True, "user": user}
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error processing data for user {user_id}: {e}\n{error_details}")
            return {"error": "An unexpected error occurred while processing data."}
    
    def register_new_user(self, name, age, preferences=None, emergency_contacts=None, medical_conditions=None):
        """Register a new user in the system.

        Args:
            name (str): User name.
            age (int): User age.
            preferences (dict, optional): User preferences.
            emergency_contacts (list, optional): Emergency contacts.
            medical_conditions (list, optional): Medical conditions.

        Returns:
            int: User ID if successful, None otherwise.
        """
        self.logger.info(f"Registering new user: {name}")

        try:
            # Add user to database
            user_id = self.db.add_user(name, age, preferences, emergency_contacts, medical_conditions)

            if user_id:
                self.logger.info(f"New user registered successfully with ID {user_id}")

                # Log the registration
                self.db.log_system_event(
                    log_type="user_registration",
                    component="main_application",
                    message=f"New user registered: {name}",
                    additional_data={"user_id": user_id}
                )

                # Create a welcome reminder
                self.db.add_reminder(
                    user_id=user_id,
                    title="Welcome to Elderly Care System",
                    reminder_type="system",
                    scheduled_time=datetime.now().isoformat(),
                    description="Thank you for joining our care system. We're here to help you stay healthy and safe.",
                    priority=2
                )

            return user_id
        except Exception as e:
            self.logger.error(f"Error adding user: {e}")
            return None
    
    def generate_user_report(self, user_id, days=7):
        """Generate a comprehensive report for a user.
        
        Args:
            user_id (int): User ID to generate report for
            days (int, optional): Number of days to include in report. Defaults to 7.
            
        Returns:
            dict: Report data
        """
        self.logger.info(f"Generating report for user {user_id} covering {days} days")
        
        try:
            # Get user information
            user = self.db.get_user(user_id)
            if not user:
                self.logger.error(f"User {user_id} not found")
                return {"error": "User not found"}
            
            # Get start and end dates
            end_date = datetime.now().isoformat()
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get health data
            health_data = self.db.get_health_data(user_id, limit=1000, start_date=start_date, end_date=end_date)
            health_stats = self.db.get_health_stats(user_id, days=days)
            
            # Get safety data
            safety_data = self.db.get_safety_data(user_id, limit=1000, start_date=start_date, end_date=end_date)
            safety_stats = self.db.get_safety_stats(user_id, days=days)
            
            # Get reminder data
            reminders = self.db.get_reminders(user_id, only_pending=False, start_date=start_date, end_date=end_date)
            reminder_stats = self.db.get_reminder_stats(user_id, days=days)
            
            # Get alert data
            alerts_query = self.db.conn.cursor()
            alerts_query.execute("""
                SELECT * FROM alerts 
                WHERE user_id = ? AND created_at BETWEEN ? AND ?
                ORDER BY created_at DESC
            """, (user_id, start_date, end_date))
            alerts = [dict(alert) for alert in alerts_query.fetchall()]
            
            # Count alerts by severity
            alert_counts = {
                "emergency": 0,
                "urgent": 0,
                "routine": 0,
                "total": len(alerts)
            }
            
            for alert in alerts:
                severity = alert.get("severity", "").lower()
                if severity in alert_counts:
                    alert_counts[severity] += 1
            
            # Compile report
            report = {
                "user": user,
                "report_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days_covered": days
                },
                "health": {
                    "stats": health_stats,
                    "data_points": len(health_data),
                    "latest_reading": health_data[0] if health_data else None
                },
                "safety": {
                    "stats": safety_stats,
                    "data_points": len(safety_data),
                    "latest_reading": safety_data[0] if safety_data else None,
                    "falls_detected": sum(1 for item in safety_data if item.get("fall_detected"))
                },
                "reminders": {
                    "stats": reminder_stats,
                    "total_reminders": len(reminders),
                    "acknowledged": sum(1 for r in reminders if r.get("is_acknowledged")),
                    "missed": sum(1 for r in reminders if not r.get("is_acknowledged") and 
                                 r.get("scheduled_time") < datetime.now().isoformat())
                },
                "alerts": {
                    "counts": alert_counts,
                    "latest_alerts": alerts[:5] if alerts else []
                },
                "generated_at": datetime.now().isoformat()
            }
            
            # Log report generation
            self.db.log_system_event(
                log_type="report_generation",
                component="main_application",
                message=f"Generated report for user {user_id}",
                additional_data={"report_days": days}
            )
            
            self.logger.info(f"Report generated successfully for user {user_id}")
            return report
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error generating report for user {user_id}: {e}\n{error_details}")
            return {"error": str(e)}
    
    def create_manual_reminder(self, user_id, title, reminder_type, scheduled_time, 
                              description=None, priority=1, recurrence=None):
        """Create a manual reminder for a user.
        
        Args:
            user_id (int): User ID
            title (str): Reminder title
            reminder_type (str): Type of reminder (medication, appointment, activity, etc.)
            scheduled_time (str): ISO format timestamp
            description (str, optional): Detailed description
            priority (int, optional): Priority level (1-3, 3 highest)
            recurrence (str, optional): Recurrence pattern
            
        Returns:
            int: Reminder ID if successful, None otherwise
        """
        self.logger.info(f"Creating manual reminder for user {user_id}: {title}")
        
        try:
            reminder_id = self.db.add_reminder(
                user_id=user_id,
                title=title,
                reminder_type=reminder_type,
                scheduled_time=scheduled_time,
                description=description,
                priority=priority,
                recurrence=recurrence
            )
            
            if reminder_id:
                self.logger.info(f"Manual reminder created with ID {reminder_id}")
                
                # Log the creation
                self.db.log_system_event(
                    log_type="reminder_creation",
                    component="main_application",
                    message=f"Manual reminder created for user {user_id}",
                    additional_data={"reminder_id": reminder_id, "title": title}
                )
            
            return reminder_id
        except Exception as e:
            self.logger.error(f"Error creating manual reminder: {e}")
            return None
    
    def create_manual_alert(self, user_id, message, severity, source="manual", recipients=None):
        """Create a manual alert for a user.
        
        Args:
            user_id (int): User ID
            message (str): Alert message
            severity (str): Alert severity (emergency, urgent, routine)
            source (str, optional): Source of the alert. Defaults to "manual".
            recipients (list, optional): List of recipient IDs. Defaults to None.
            
        Returns:
            int: Alert ID if successful, None otherwise
        """
        self.logger.info(f"Creating manual alert for user {user_id}: {severity}")
        
        try:
            # Create the alert
            alert_id = self.alert_system.create_alert(
                user_id=user_id,
                message=message,
                severity=severity,
                source_agent=source
            )
            
            # Add recipients if provided
            if alert_id and recipients:
                for recipient in recipients:
                    if isinstance(recipient, dict) and 'id' in recipient and 'type' in recipient:
                        self.db.add_alert_recipient(
                            alert_id=alert_id,
                            recipient_id=recipient['id'],
                            recipient_type=recipient['type']
                        )
            
            if alert_id:
                self.logger.info(f"Manual alert created with ID {alert_id}")
                
                # Log the creation
                self.db.log_system_event(
                    log_type="alert_creation",
                    component="main_application",
                    message=f"Manual alert created for user {user_id}",
                    additional_data={"alert_id": alert_id, "severity": severity}
                )
            
            return alert_id
        except Exception as e:
            self.logger.error(f"Error creating manual alert: {e}")
            return None
    
    def get_pending_items(self, user_id: int) -> List[Dict]:
        """
        Get all pending items (unacknowledged reminders and alerts) for a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            list: List of pending items
        """
        try:
            pending_items = []
            
            # Get unacknowledged reminders
            reminders = self.db.get_reminders(user_id=user_id, only_pending=True)
            for reminder in reminders:
                pending_items.append({
                    "type": "reminder",
                    "id": reminder["id"],
                    "title": reminder["title"],
                    "description": reminder["description"],
                    "scheduled_time": reminder["scheduled_time"],
                    "priority": reminder["priority"]
                })
            
            # Get unacknowledged alerts
            alerts = self.db.get_active_alerts(user_id=user_id)
            for alert in alerts:
                pending_items.append({
                    "type": "alert",
                    "id": alert["id"],
                    "message": alert["message"],
                    "severity": alert["severity"],
                    "created_at": alert["created_at"]
                })
            
            return pending_items
            
        except Exception as e:
            self.logger.error(f"Error getting pending items: {e}")
            return []
    
    def perform_system_maintenance(self):
        """Perform system maintenance tasks.
        
        Returns:
            dict: Results of maintenance operations
        """
        self.logger.info("Starting system maintenance")
        results = {
            "started_at": datetime.now().isoformat(),
            "operations": {},
            "errors": []
        }
        
        try:
            # 1. Clean up old data based on retention policy
            retention_days = self.config.get("data_retention_days", 90)
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
            
            cursor = self.db.conn.cursor()
            
            # Clean health data
            cursor.execute("DELETE FROM health_data WHERE timestamp < ?", (cutoff_date,))
            results["operations"]["health_data_cleanup"] = cursor.rowcount
            
            # Clean safety data
            cursor.execute("DELETE FROM safety_data WHERE timestamp < ?", (cutoff_date,))
            results["operations"]["safety_data_cleanup"] = cursor.rowcount
            
            # Clean old acknowledged reminders
            cursor.execute("""
                DELETE FROM reminders 
                WHERE is_acknowledged = 1 AND acknowledged_time < ?
            """, (cutoff_date,))
            results["operations"]["reminders_cleanup"] = cursor.rowcount
            
            # Clean old acknowledged alerts
            cursor.execute("""
                DELETE FROM alerts 
                WHERE is_acknowledged = 1 AND acknowledged_time < ?
            """, (cutoff_date,))
            results["operations"]["alerts_cleanup"] = cursor.rowcount
            
            # Clean old system logs
            cursor.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff_date,))
            results["operations"]["logs_cleanup"] = cursor.rowcount
            
            self.db.conn.commit()
            
            # 2. Vacuum the database to reclaim space
            self.db.conn.execute("VACUUM")
            results["operations"]["database_vacuum"] = True
            
            # Log the maintenance
            self.db.log_system_event(
                log_type="maintenance",
                component="main_application",
                message="System maintenance performed",
                additional_data=results["operations"]
            )
            
            results["completed_at"] = datetime.now().isoformat()
            self.logger.info("System maintenance completed successfully")
            
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error during system maintenance: {e}\n{error_details}")
            results["errors"].append(str(e))
            results["completed_at"] = datetime.now().isoformat()
            
            # Log the error
            self.db.log_system_event(
                log_type="error",
                component="main_application",
                message="Error during system maintenance",
                additional_data={"error": str(e)}
            )
        
        return results
    
    def get_system_stats(self):
        """Get system statistics.
        
        Returns:
            dict: System statistics
        """
        self.logger.info("Retrieving system statistics")
        
        try:
            cursor = self.db.conn.cursor()
            stats = {}
            
            # Count users
            cursor.execute("SELECT COUNT(*) FROM users")
            stats["total_users"] = cursor.fetchone()[0]
            
            # Count caregivers
            cursor.execute("SELECT COUNT(*) FROM caregivers")
            stats["total_caregivers"] = cursor.fetchone()[0]
            
            # Count active alerts
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE is_acknowledged = 0")
            stats["active_alerts"] = cursor.fetchone()[0]
            
            # Count today's reminders
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("""
                SELECT COUNT(*) FROM reminders 
                WHERE date(scheduled_time) = ?
            """, (today,))
            stats["todays_reminders"] = cursor.fetchone()[0]
            
            # Count data entries in last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            
            cursor.execute("""
                SELECT COUNT(*) FROM health_data 
                WHERE timestamp > ?
            """, (yesterday,))
            stats["health_entries_24h"] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM safety_data 
                WHERE timestamp > ?
            """, (yesterday,))
            stats["safety_entries_24h"] = cursor.fetchone()[0]
            
            # Count total data volume
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM health_data) AS health_count,
                    (SELECT COUNT(*) FROM safety_data) AS safety_count,
                    (SELECT COUNT(*) FROM reminders) AS reminder_count,
                    (SELECT COUNT(*) FROM alerts) AS alert_count,
                    (SELECT COUNT(*) FROM system_logs) AS log_count
            """)
            
            data_counts = cursor.fetchone()
            stats["data_volume"] = {
                "health_data": data_counts[0],
                "safety_data": data_counts[1],
                "reminders": data_counts[2],
                "alerts": data_counts[3],
                "system_logs": data_counts[4],
                "total_entries": sum(data_counts)
            }
            
            # Get database size
            stats["database_size_kb"] = os.path.getsize(self.db.db_path) / 1024
            
            # System uptime could be added if running as a service
            
            stats["retrieved_at"] = datetime.now().isoformat()
            self.logger.info("System statistics retrieved successfully")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error retrieving system statistics: {e}")
            return {"error": str(e)}
    
    def start_autonomous_operations(self):
        """Start autonomous operations for all agents."""
        self.logger.info("Starting autonomous agent operations")
        try:
            # Schedule periodic checks for all agents
            self.agent_coordinator.schedule_periodic_checks()
            self.logger.info("Autonomous operations started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start autonomous operations: {e}")
            return False
            
    def stop_autonomous_operations(self):
        """Stop all autonomous agent operations."""
        self.logger.info("Stopping autonomous agent operations")
        try:
            # Disable periodic checks
            if hasattr(self.agent_coordinator, 'stop_autonomous_operations'):
                self.agent_coordinator.stop_autonomous_operations = True
                self.logger.info("Autonomous operations stopped successfully")
                return True
            else:
                self.logger.warning("Agent coordinator does not support stopping autonomous operations")
                return False
        except Exception as e:
            self.logger.error(f"Failed to stop autonomous operations: {e}")
            return False
    
    def close(self):
        """Clean up resources when shutting down the system."""
        self.logger.info("Shutting down Elderly Care System")
        try:
            # Stop autonomous operations first
            self.stop_autonomous_operations()
            
            # Close database connection
            if hasattr(self, 'db'):
                self.db.close()
                self.logger.info("Database connection closed")
            
            self.logger.info("Elderly Care System shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {e}")
            raise

    def process_health_data(self, user_id: int, data: dict) -> dict:
        """
        Process health data for a specific user and generate insights or alerts.
        
        Args:
            user_id (int): The ID of the user to process health data for
            data (dict): The health data to process
            
        Returns:
            dict: Results of processing the health data
        """
        self.logger.info(f"Processing health data for user {user_id}")
        
        try:
            # Get the user details
            user = self.db.get_user(user_id)
            if not user:
                self.logger.error(f"User {user_id} not found")
                return {"error": "User not found"}
            
            age = user.get('age', 70)  # Default to elderly age if not specified
            user_sex = user.get('sex', 'unknown')  # Get sex for appropriate thresholds
                
            results = {
                "anomalies_detected": False,
                "issues": [],
                "actions_taken": [],
                "metrics": {},
                "status_summary": {}
            }
            
            # Check if data contains anomalies
            heart_rate = data.get('heart_rate')
            blood_pressure_systolic = data.get('blood_pressure_systolic')
            blood_pressure_diastolic = data.get('blood_pressure_diastolic')
            temperature = data.get('temperature')
            oxygen_saturation = data.get('oxygen_saturation')
            glucose_level = data.get('glucose_level')
            
            # Store metrics for the result with status
            metrics_status = {}
            
            # Heart Rate thresholds - adjusted for elderly (more lenient)
            if heart_rate is not None:
                results["metrics"]["heart_rate"] = f"{heart_rate} BPM"
                
                if heart_rate < 40:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Critically low heart rate: {heart_rate} BPM",
                        "severity": "high"
                    })
                elif heart_rate < 50:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Low heart rate: {heart_rate} BPM",
                        "severity": "medium"
                    })
                elif heart_rate > 120:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Critically elevated heart rate: {heart_rate} BPM",
                        "severity": "high"
                    })
                elif heart_rate > 100:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Elevated heart rate: {heart_rate} BPM",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["heart_rate"] = status
            
            # Temperature thresholds (Celsius)
            if temperature is not None:
                results["metrics"]["temperature"] = f"{temperature}°C"
                
                if temperature < 35.0:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypothermia detected: {temperature}°C",
                        "severity": "high"
                    })
                elif temperature < 36.0:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Below normal temperature: {temperature}°C",
                        "severity": "medium"
                    })
                elif temperature > 39.0:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"High fever detected: {temperature}°C",
                        "severity": "high"
                    })
                elif temperature > 37.8:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Fever detected: {temperature}°C",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["temperature"] = status
                
            # Oxygen saturation thresholds
            if oxygen_saturation is not None:
                results["metrics"]["oxygen_saturation"] = f"{oxygen_saturation}%"
                
                if oxygen_saturation < 88:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Severe hypoxemia detected: {oxygen_saturation}%",
                        "severity": "high"
                    })
                elif oxygen_saturation < 92:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Mild hypoxemia detected: {oxygen_saturation}%",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["oxygen_saturation"] = status
                
            # Blood pressure thresholds
            if blood_pressure_systolic is not None and blood_pressure_diastolic is not None:
                results["metrics"]["blood_pressure"] = f"{blood_pressure_systolic}/{blood_pressure_diastolic} mmHg"
                
                # Determine overall BP status (worst of systolic/diastolic)
                systolic_status = "normal"
                diastolic_status = "normal"
                
                # Systolic BP evaluation
                if blood_pressure_systolic < 90:
                    systolic_status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypotension detected: Systolic BP {blood_pressure_systolic} mmHg",
                        "severity": "high"
                    })
                elif blood_pressure_systolic < 100:
                    systolic_status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Low systolic BP: {blood_pressure_systolic} mmHg",
                        "severity": "medium"
                    })
                elif blood_pressure_systolic >= 180:
                    systolic_status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertensive crisis: Systolic BP {blood_pressure_systolic} mmHg",
                        "severity": "high"
                    })
                elif blood_pressure_systolic >= 140:
                    systolic_status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertension detected: Systolic BP {blood_pressure_systolic} mmHg",
                        "severity": "medium"
                    })
                
                # Diastolic BP evaluation
                if blood_pressure_diastolic < 60:
                    diastolic_status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Low diastolic BP: {blood_pressure_diastolic} mmHg",
                        "severity": "medium"
                    })
                elif blood_pressure_diastolic >= 120:
                    diastolic_status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertensive crisis: Diastolic BP {blood_pressure_diastolic} mmHg",
                        "severity": "high"
                    })
                elif blood_pressure_diastolic >= 90:
                    diastolic_status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertension detected: Diastolic BP {blood_pressure_diastolic} mmHg",
                        "severity": "medium"
                    })
                
                # Overall BP status is the more severe of the two
                if systolic_status == "critical" or diastolic_status == "critical":
                    status = "critical"
                elif systolic_status == "moderate" or diastolic_status == "moderate":
                    status = "moderate"
                else:
                    status = "normal"
                    
                metrics_status["blood_pressure"] = status
                
            elif blood_pressure_systolic is not None:
                results["metrics"]["blood_pressure_systolic"] = f"{blood_pressure_systolic} mmHg"
                
                if blood_pressure_systolic < 90:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypotension detected: Systolic BP {blood_pressure_systolic} mmHg",
                        "severity": "high"
                    })
                elif blood_pressure_systolic < 100:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Low systolic BP: {blood_pressure_systolic} mmHg",
                        "severity": "medium"
                    })
                elif blood_pressure_systolic >= 180:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertensive crisis: Systolic BP {blood_pressure_systolic} mmHg",
                        "severity": "high"
                    })
                elif blood_pressure_systolic >= 140:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertension detected: Systolic BP {blood_pressure_systolic} mmHg",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["blood_pressure_systolic"] = status
                
            elif blood_pressure_diastolic is not None:
                results["metrics"]["blood_pressure_diastolic"] = f"{blood_pressure_diastolic} mmHg"
                
                if blood_pressure_diastolic < 60:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Low diastolic BP: {blood_pressure_diastolic} mmHg",
                        "severity": "medium"
                    })
                elif blood_pressure_diastolic >= 120:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertensive crisis: Diastolic BP {blood_pressure_diastolic} mmHg",
                        "severity": "high"
                    })
                elif blood_pressure_diastolic >= 90:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypertension detected: Diastolic BP {blood_pressure_diastolic} mmHg",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["blood_pressure_diastolic"] = status
                
            # Glucose level thresholds (mg/dL)
            if glucose_level is not None:
                results["metrics"]["glucose_level"] = f"{glucose_level} mg/dL"
                
                if glucose_level < 70:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hypoglycemia detected: Glucose {glucose_level} mg/dL",
                        "severity": "medium"
                    })
                elif glucose_level < 55:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Severe hypoglycemia detected: Glucose {glucose_level} mg/dL",
                        "severity": "high"
                    })
                elif glucose_level > 300:
                    status = "critical"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Severe hyperglycemia detected: Glucose {glucose_level} mg/dL",
                        "severity": "high"
                    })
                elif glucose_level > 180:
                    status = "moderate"
                    results["anomalies_detected"] = True
                    results["issues"].append({
                        "message": f"Hyperglycemia detected: Glucose {glucose_level} mg/dL",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["glucose_level"] = status
                
            # Add status summary to results
            results["status_summary"] = metrics_status
            
            # If the user has the health monitoring agent, use it
            if hasattr(self.agent_coordinator, 'health_monitoring_agent'):
                agent = self.agent_coordinator.health_monitoring_agent
                if hasattr(agent, 'analyze_health_data'):
                    agent_results = agent.analyze_health_data(user_id, data)
                    # Merge results
                    if agent_results:
                        results["agent_analysis"] = agent_results
                        
                        if agent_results.get('anomalies_detected'):
                            results["anomalies_detected"] = True
                            
                        if agent_results.get('insights'):
                            for insight in agent_results.get('insights', []):
                                if isinstance(insight, str):
                                    results["issues"].append({
                                        "message": insight,
                                        "severity": "medium"
                                    })
                                else:
                                    results["issues"].append(insight)
                            
                        if agent_results.get('actions_taken'):
                            results["actions_taken"].extend(agent_results.get('actions_taken', []))
                            
            # If anomalies were detected, create an alert
            if results["anomalies_detected"]:
                alert_messages = []
                alert_severity = "info"
                
                # Extract alerts from issues, determining the highest severity
                for issue in results["issues"]:
                    if isinstance(issue, dict) and "message" in issue:
                        alert_messages.append(issue["message"])
                        if issue.get("severity") == "high":
                            alert_severity = "critical"
                        elif issue.get("severity") == "medium" and alert_severity == "info":
                            alert_severity = "warning"
                    elif isinstance(issue, str):
                        alert_messages.append(issue)
                
                # Create alert message
                alert_message = "Health anomaly detected: " + ", ".join(alert_messages)
                
                # Add alert to database
                self.db.add_alert(
                    user_id=user_id,
                    message=alert_message,
                    severity=alert_severity,
                    source_agent="health_monitoring_agent",
                    additional_data=json.dumps(data)
                )
                
                results["actions_taken"].append("Created health alert for detected anomalies")
                self.logger.warning(f"Health alert created for user {user_id}: {alert_message}")
            
            # Add recommendations if anomalies detected
            if results["anomalies_detected"]:
                results["recommendations"] = [
                    "Contact healthcare provider immediately for any critical values",
                    "Monitor and document symptoms associated with abnormal readings",
                    "Ensure proper medication adherence and timing",
                    "Maintain adequate hydration and follow dietary guidelines",
                    "Verify measurement devices are calibrated and working properly"
                ]
                
                # Add specific recommendations based on detected issues
                has_blood_pressure_issue = any("blood pressure" in issue["message"].lower() 
                                               for issue in results["issues"] 
                                               if isinstance(issue, dict) and "message" in issue)
                
                has_glucose_issue = any("glucose" in issue["message"].lower() 
                                        for issue in results["issues"] 
                                        if isinstance(issue, dict) and "message" in issue)
                
                if has_blood_pressure_issue:
                    results["recommendations"].append("Limit sodium intake and follow DASH diet principles")
                    
                if has_glucose_issue:
                    results["recommendations"].append("Monitor carbohydrate intake and follow prescribed diabetes management plan")
            
            # Add overall health status assessment
            if results["anomalies_detected"]:
                # Determine overall severity
                has_critical = any(issue.get("severity") == "high" 
                                  for issue in results["issues"] 
                                  if isinstance(issue, dict) and "severity" in issue)
                
                if has_critical:
                    results["overall_status"] = "critical"
                else:
                    results["overall_status"] = "moderate"
            else:
                results["overall_status"] = "normal"
            
            # Log the processing complete
            self.logger.info(f"Completed processing health data for user {user_id}")
            
            return results
            
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error processing health data for user {user_id}: {e}\n{error_details}")
            return {"error": f"An error occurred while processing health data: {str(e)}"}
            
    def process_safety_data(self, user_id: int, data: dict) -> dict:
        """
        Process safety data for a specific user and generate insights or alerts.
        
        Args:
            user_id (int): The ID of the user to process safety data for
            data (dict): The safety data to process
            
        Returns:
            dict: Results of processing the safety data
        """
        self.logger.info(f"Processing safety data for user {user_id}")
        
        try:
            # Get the user details
            user = self.db.get_user(user_id)
            if not user:
                self.logger.error(f"User {user_id} not found")
                return {"error": "User not found"}
                
            results = {
                "issues_detected": False,
                "issues": [],
                "actions_taken": [],
                "metrics": {},
                "status_summary": {}
            }
            
            # Extract safety metrics
            fall_detected = bool(data.get('fall_detected', False))
            location = data.get('location')
            movement_type = data.get('movement_type')
            activity_level = data.get('activity_level')
            time_inactive = data.get('time_inactive')
            risk_level = data.get('risk_level')
            
            # Store metrics for the result
            metrics_status = {}
            
            # Location assessment
            if location:
                results["metrics"]["location"] = location
                metrics_status["location"] = "normal"
                
                # Check if bathroom location with extended inactivity
                if location.lower() == 'bathroom' and time_inactive and time_inactive > 60:
                    metrics_status["location"] = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"Extended bathroom stay detected: {time_inactive} minutes",
                        "severity": "medium"
                    })
            
            # Movement type assessment
            if movement_type:
                results["metrics"]["movement_type"] = movement_type
                
                if movement_type.lower() == 'unknown':
                    status = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": "Movement could not be determined",
                        "severity": "medium"
                    })
                elif movement_type.lower() == 'lying_down' and location and location.lower() not in ['bedroom', 'living_room']:
                    status = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"User lying down in {location}",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["movement_type"] = status
            
            # Activity level assessment
            if activity_level:
                results["metrics"]["activity_level"] = activity_level
                
                if activity_level.lower() == 'none':
                    status = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": "No activity detected",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["activity_level"] = status
            
            # Fall detection (critical safety issue)
            if fall_detected is not None:
                results["metrics"]["fall_detected"] = "Yes" if fall_detected else "No"
                
                if fall_detected:
                    status = "critical"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": "Fall detected - immediate assistance required",
                        "severity": "high"
                    })
                else:
                    status = "normal"
                    
                metrics_status["fall_detected"] = status
            
            # Inactivity assessment
            if time_inactive is not None:
                results["metrics"]["time_inactive"] = f"{time_inactive} minutes"
                
                if time_inactive > 480:  # 8 hours
                    status = "critical"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"Critical inactivity detected: {time_inactive} minutes without movement",
                        "severity": "high"
                    })
                elif time_inactive > 240:  # 4 hours
                    status = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"Extended inactivity detected: {time_inactive} minutes",
                        "severity": "medium"
                    })
                elif time_inactive > 120 and (activity_level and activity_level.lower() == 'none'):
                    status = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"Inactivity with no movement: {time_inactive} minutes",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["time_inactive"] = status
            
            # Risk level assessment
            if risk_level:
                results["metrics"]["risk_level"] = risk_level
                
                if risk_level.lower() == 'high':
                    status = "critical"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"High safety risk detected",
                        "severity": "high"
                    })
                elif risk_level.lower() == 'medium':
                    status = "moderate"
                    results["issues_detected"] = True
                    results["issues"].append({
                        "message": f"Medium safety risk detected",
                        "severity": "medium"
                    })
                else:
                    status = "normal"
                    
                metrics_status["risk_level"] = status
                
            # Add status summary to results
            results["status_summary"] = metrics_status
                
            # If the user has the safety monitoring agent, use it
            if hasattr(self.agent_coordinator, 'safety_monitoring_agent'):
                agent = self.agent_coordinator.safety_monitoring_agent
                if hasattr(agent, 'analyze_safety_data'):
                    agent_results = agent.analyze_safety_data(user_id, data)
                    # Merge results
                    if agent_results:
                        results["agent_analysis"] = agent_results
                        
                        if agent_results.get('issues_detected'):
                            results["issues_detected"] = True
                            
                        if agent_results.get('insights'):
                            for insight in agent_results.get('insights', []):
                                if isinstance(insight, str):
                                    results["issues"].append({
                                        "message": insight,
                                        "severity": "medium"
                                    })
                                else:
                                    results["issues"].append(insight)
                            
                        if agent_results.get('actions_taken'):
                            results["actions_taken"].extend(agent_results.get('actions_taken', []))
                            
            # If issues were detected, create an alert
            if results["issues_detected"]:
                alert_messages = []
                alert_severity = "info"
                
                # Extract alerts from issues, determining the highest severity
                for issue in results["issues"]:
                    if isinstance(issue, dict) and "message" in issue:
                        alert_messages.append(issue["message"])
                        if issue.get("severity") == "high":
                            alert_severity = "critical"
                        elif issue.get("severity") == "medium" and alert_severity == "info":
                            alert_severity = "warning"
                    elif isinstance(issue, str):
                        alert_messages.append(issue)
                
                # Create alert message
                alert_message = "Safety issue detected: " + ", ".join(alert_messages)
                
                # Add alert to database
                self.db.add_alert(
                    user_id=user_id,
                    message=alert_message,
                    severity=alert_severity,
                    source_agent="safety_monitoring_agent",
                    additional_data=json.dumps(data)
                )
                
                results["actions_taken"].append("Created safety alert for detected issues")
                self.logger.warning(f"Safety alert created for user {user_id}: {alert_message}")
            
            # Add recommendations based on detected issues
            if results["issues_detected"]:
                results["recommendations"] = [
                    "Check on the user immediately for critical safety situations",
                    "Maintain clear pathways and adequate lighting",
                    "Ensure emergency call systems are operational and accessible",
                    "Review fall prevention measures in the environment"
                ]
                
                # Add specific recommendations based on detected issues
                if fall_detected:
                    results["recommendations"] = [
                        "URGENT: Check on the user immediately - fall detected",
                        "Assess for injuries and call emergency services if needed",
                        "Do not move the person if severe injury is suspected",
                        "Document the incident and review fall prevention measures"
                    ]
                elif time_inactive and time_inactive > 240:
                    results["recommendations"].append("Schedule regular movement or position changes")
                    results["recommendations"].append("Assess for potential mobility issues or discomfort")
            
            # Add overall safety status assessment
            if results["issues_detected"]:
                # Determine overall severity
                has_critical = any(issue.get("severity") == "high" 
                                  for issue in results["issues"] 
                                  if isinstance(issue, dict) and "severity" in issue)
                
                if has_critical:
                    results["overall_status"] = "critical"
                else:
                    results["overall_status"] = "moderate"
                
            # Log the processing complete
            self.logger.info(f"Completed processing safety data for user {user_id}")
            
            return results
            
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Error processing safety data for user {user_id}: {e}\n{error_details}")
            return {"error": f"An error occurred while processing safety data: {str(e)}"}

def main():
    """Main function to run the system."""
    logger.info("Starting Elderly Care Multi-Agent System")
    system = None

    try:
        # Initialize the system
        system = ElderlyCareSys()

        # Production-ready: Replace with API calls or scheduled tasks
        logger.info("Elderly Care System is running. Waiting for tasks...")
        
        while True:
            try:
                # Example: Perform periodic maintenance
                system.perform_system_maintenance()
                time.sleep(3600)  # Sleep for 1 hour
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Sleep for 1 minute before retrying

    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        if system:
            system.close()
        logger.info("Elderly Care Multi-Agent System stopped")

if __name__ == "__main__":
    main()