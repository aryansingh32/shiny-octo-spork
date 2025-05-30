import os
import sys
import json
import uuid
import logging
from datetime import datetime
import sqlite3
import pyttsx3
from logging.handlers import RotatingFileHandler
import traceback
import time
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend.env')
load_dotenv(env_path)

# Setup logging with rotation and better formatting
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, "alerts.log"),
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AlertSystem")

class Database:
    def __init__(self, db_path):
        """Initialize database connection with proper error handling."""
        try:
            abs_db_path = os.path.abspath(db_path)
            logger.info(f"Connecting to database at: {abs_db_path}")
            
            self.connection = sqlite3.connect(abs_db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            logger.info("Database connected successfully")
            
            # Verify connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            logger.info(f"Found {len(tables)} tables in the database")
            
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {str(e)}")
            raise

    def close(self):
        """Safely close the database connection."""
        if hasattr(self, 'connection') and self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")

class AlertSystem:
    """
    Handles all alerts generated by the multi-agent system.
    
    This class is responsible for:
    - Creating, categorizing, and storing alerts
    - Routing alerts to appropriate stakeholders
    - Tracking alert statuses and acknowledgments
    - Retrieving active and historical alerts
    """
    
    def __init__(self, db_connection=None, db_path=None):
        """Initialize the AlertSystem with improved error handling."""
        logger.info("Initializing AlertSystem")
        
        try:
            # Use DATABASE_PATH from environment variables if db_path is not provided
            if db_path is None:
                db_path = os.getenv("DATABASE_PATH", "elderly_care.db")
            
            # Initialize db_connection properly
            if db_connection is None:
                self.db = Database(db_path)
                self.db_connection = self.db.conn
            elif isinstance(db_connection, sqlite3.Connection):
                self.db = None
                self.db_connection = db_connection
            else:
                raise TypeError("db_connection must be a sqlite3.Connection or None")
            
            # Create alert tables if they don't exist
            self._create_tables()
            
            # Define severity levels
            self.severity_levels = {
                "emergency": {"priority": 3, "color": "red", "response_time": "immediate"},
                "urgent": {"priority": 2, "color": "orange", "response_time": "within 30 minutes"},
                "routine": {"priority": 1, "color": "yellow", "response_time": "within 24 hours"},
                "info": {"priority": 0, "color": "blue", "response_time": "no specific timeframe"}
            }
            
            # Define stakeholder types
            self.stakeholder_types = [
                "caregiver", 
                "family_member", 
                "healthcare_provider", 
                "emergency_services"
            ]
            
            logger.info("AlertSystem initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AlertSystem: {e}")
            raise

    def __del__(self):
        """Cleanup method to ensure resources are properly released."""
        if hasattr(self, 'db') and self.db:
            self.db.close()

    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor operations."""
        cursor = self.db_connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _create_tables(self):
        """Create necessary database tables with improved error handling."""
        try:
            with self._get_cursor() as cursor:
                # Create alerts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT 0,
                    acknowledged_at TIMESTAMP,
                    acknowledged_by TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_at TIMESTAMP,
                    resolved_by TEXT,
                    additional_data TEXT
                )
                ''')
                
                # Create alert_recipients table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_recipients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT,
                    recipient_id TEXT NOT NULL,
                    recipient_type TEXT NOT NULL,
                    notified BOOLEAN DEFAULT 0,
                    notified_at TIMESTAMP,
                    FOREIGN KEY (alert_id) REFERENCES alerts (alert_id)
                )
                ''')
                
                self.db_connection.commit()
                logger.info("Alert tables created successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {str(e)}")
            self.db_connection.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating tables: {str(e)}")
            self.db_connection.rollback()
            raise

    def create_alert(self, user_id, message, severity, source_agent, additional_data=None, recipients=None):
        """Create and store a new alert with improved error handling."""
        try:
            # Generate unique alert ID
            alert_id = str(uuid.uuid4())
            
            # Validate severity
            if severity not in self.severity_levels:
                logger.warning(f"Invalid severity '{severity}', defaulting to 'info'")
                severity = "info"
            
            # Prepare additional data for storage
            additional_data_json = json.dumps(additional_data) if additional_data else None
            
            with self._get_cursor() as cursor:
                # Insert alert into database
                cursor.execute('''
                INSERT INTO alerts 
                (alert_id, user_id, message, severity, source_agent, created_at, additional_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id, 
                    user_id, 
                    message, 
                    severity, 
                    source_agent, 
                    datetime.now().isoformat(), 
                    additional_data_json
                ))
                
                # Insert recipients if provided
                if recipients:
                    for recipient in recipients:
                        if 'type' not in recipient or recipient['type'] not in self.stakeholder_types:
                            logger.warning(f"Invalid recipient type: {recipient.get('type')}")
                            continue
                        
                        if 'id' not in recipient:
                            logger.warning("Recipient missing ID field")
                            continue
                        
                        cursor.execute('''
                        INSERT INTO alert_recipients 
                        (alert_id, recipient_id, recipient_type)
                        VALUES (?, ?, ?)
                        ''', (
                            alert_id,
                            recipient['id'],
                            recipient['type']
                        ))
                else:
                    # Auto-assign recipients based on severity
                    self._auto_assign_recipients(cursor, alert_id, user_id, severity)
                
                self.db_connection.commit()
                logger.info(f"Created alert {alert_id} for user {user_id} with severity {severity}")
                
                # Trigger notification system
                self._send_notifications(alert_id, severity)
                
                return alert_id
                
        except sqlite3.Error as e:
            logger.error(f"Database error creating alert: {str(e)}")
            self.db_connection.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating alert: {str(e)}")
            self.db_connection.rollback()
            raise
    
    def _auto_assign_recipients(self, cursor, alert_id, user_id, severity):
        """
        Automatically assign recipients based on alert severity.
        
        Args:
            cursor: Database cursor
            alert_id (str): ID of the alert
            user_id (str): ID of the user
            severity (str): Alert severity
        """
        try:
            # Get user's caregivers, family, and healthcare providers (would normally query user_stakeholders table)
            # For now, we'll use dummy data
            
            # Emergency alerts go to everyone
            if severity == "emergency":
                # Emergency services
                cursor.execute('''
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type)
                VALUES (?, ?, ?)
                ''', (alert_id, "emergency_service_1", "emergency_services"))
                
                # Primary caregiver
                cursor.execute('''
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type)
                VALUES (?, ?, ?)
                ''', (alert_id, f"caregiver_{user_id}", "caregiver"))
                
                # Family members
                cursor.execute('''
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type)
                VALUES (?, ?, ?)
                ''', (alert_id, f"family_{user_id}_1", "family_member"))
            
            # Urgent alerts go to caregiver and family
            elif severity == "urgent":
                # Primary caregiver
                cursor.execute('''
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type)
                VALUES (?, ?, ?)
                ''', (alert_id, f"caregiver_{user_id}", "caregiver"))
                
                # Family members
                cursor.execute('''
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type)
                VALUES (?, ?, ?)
                ''', (alert_id, f"family_{user_id}_1", "family_member"))
            
            # Routine alerts go to caregiver
            elif severity == "routine":
                # Primary caregiver
                cursor.execute('''
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type)
                VALUES (?, ?, ?)
                ''', (alert_id, f"caregiver_{user_id}", "caregiver"))
            
            # Info alerts are stored but not sent to anyone
            else:
                pass
            
        except sqlite3.Error as e:
            logger.error(f"Error auto-assigning recipients: {str(e)}")
            raise
    
    def _send_notifications(self, alert_id, severity):
        """
        Send notifications for an alert (placeholder for external notification service).
        
        Args:
            alert_id (str): ID of the alert
            severity (str): Alert severity
        """
        logger.info(f"Sending notifications for alert {alert_id} with severity {severity}")
        
        # Example: Integrate with an external notification service
        # For now, simulate notifications by updating the database
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
            UPDATE alert_recipients
            SET notified = 1, notified_at = ?
            WHERE alert_id = ?
            ''', (datetime.now().isoformat(), alert_id))
            
            self.db_connection.commit()
            logger.info(f"Marked notifications as sent for alert {alert_id}")
            
        except sqlite3.Error as e:
            logger.error(f"Error updating notification status: {str(e)}")
            self.db_connection.rollback()
    
    def get_active_alerts(self, user_id=None, severity=None, limit=100):
        """
        Retrieve current active (unresolved) alerts.
        
        Args:
            user_id (str): Optional user ID to filter alerts
            severity (str): Optional severity level to filter alerts
            limit (int): Maximum number of alerts to return
            
        Returns:
            list: List of active alerts
        """
        try:
            cursor = self.db_connection.cursor()
            
            query = "SELECT * FROM alerts WHERE resolved = 0"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            # Use a more reliable way to get column names
            columns = [column[0] for column in cursor.description]
            alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # For consistency with the API, add an alert_id field that matches the id
            for alert in alerts:
                if 'id' in alert and 'alert_id' not in alert:
                    alert['alert_id'] = alert['id']
                
                # Parse JSON stored in additional_data
                if alert.get('additional_data'):
                    try:
                        alert['additional_data'] = json.loads(alert['additional_data'])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse additional_data for alert {alert.get('id')}")
            
            logger.info(f"Retrieved {len(alerts)} active alerts")
            return alerts
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving active alerts: {str(e)}")
            raise
    
    def get_active_alerts_count(self) -> dict:
        """Get counts of active alerts by severity and type."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN severity = 'emergency' THEN 1 ELSE 0 END) as high,
                    SUM(CASE WHEN severity = 'urgent' THEN 1 ELSE 0 END) as medium,
                    SUM(CASE WHEN severity = 'routine' THEN 1 ELSE 0 END) as low,
                    SUM(CASE WHEN severity = 'info' THEN 1 ELSE 0 END) as info,
                    SUM(CASE WHEN source_agent = 'health_monitoring' THEN 1 ELSE 0 END) as health,
                    SUM(CASE WHEN source_agent = 'safety_monitoring' THEN 1 ELSE 0 END) as safety,
                    SUM(CASE WHEN source_agent = 'system' THEN 1 ELSE 0 END) as system
                FROM alerts
                WHERE is_acknowledged = 0
            """)
            result = cursor.fetchone()
            return {
                'total': result['total'] if result else 0,
                'bySeverity': {
                    'high': result['high'] if result else 0,
                    'medium': result['medium'] if result else 0,
                    'low': result['low'] if result else 0,
                    'info': result['info'] if result else 0
                },
                'byType': {
                    'health': result['health'] if result else 0,
                    'safety': result['safety'] if result else 0,
                    'system': result['system'] if result else 0
                }
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting active alerts count: {e}")
            raise
    
    def acknowledge_alert(self, alert_id, acknowledged_by):
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_id (str): ID of the alert to acknowledge
            acknowledged_by (str): ID of the person acknowledging the alert
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Update alert status
            cursor.execute('''
            UPDATE alerts
            SET acknowledged = 1, acknowledged_at = ?, acknowledged_by = ?
            WHERE alert_id = ?
            ''', (datetime.now().isoformat(), acknowledged_by, alert_id))
            
            if cursor.rowcount == 0:
                logger.warning(f"No alert found with ID {alert_id}")
                return False
            
            self.db_connection.commit()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
            self.db_connection.rollback()
            return False
    
    def resolve_alert(self, alert_id, resolved_by, resolution_notes=None):
        """
        Mark an alert as resolved.
        
        Args:
            alert_id (str): ID of the alert to resolve
            resolved_by (str): ID of the person resolving the alert
            resolution_notes (str): Optional notes about resolution
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Get the current additional_data
            cursor.execute("SELECT additional_data FROM alerts WHERE alert_id = ?", (alert_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No alert found with ID {alert_id}")
                return False
            
            # Update additional_data with resolution notes if provided
            additional_data = {}
            if result[0]:
                try:
                    additional_data = json.loads(result[0])
                    if not isinstance(additional_data, dict):
                        additional_data = {}
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse additional_data for alert {alert_id}")
                    additional_data = {}
            
            if resolution_notes:
                additional_data["resolution_notes"] = resolution_notes
            
            # Update alert status
            cursor.execute('''
            UPDATE alerts
            SET resolved = 1, resolved_at = ?, resolved_by = ?, additional_data = ?
            WHERE alert_id = ?
            ''', (
                datetime.now().isoformat(), 
                resolved_by, 
                json.dumps(additional_data), 
                alert_id
            ))
            
            self.db_connection.commit()
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error resolving alert: {str(e)}")
            self.db_connection.rollback()
            return False
    
    def get_alert_history(self, user_id, days=7, include_resolved=True, limit=100):
        """
        Retrieve alert history for a user.
        
        Args:
            user_id (str): ID of the user
            days (int): Number of days of history to include
            include_resolved (bool): Whether to include resolved alerts
            limit (int): Maximum number of alerts to return
            
        Returns:
            list: List of historical alerts
        """
        try:
            cursor = self.db_connection.cursor()
            
            query = f'''
            SELECT * FROM alerts 
            WHERE user_id = ? 
            AND created_at >= datetime('now', '-{days} days')
            '''
            
            if not include_resolved:
                query += " AND resolved = 0"
            
            query += " ORDER BY created_at DESC LIMIT ?"
            
            cursor.execute(query, (user_id, limit))
            columns = [column[0] for column in cursor.description]
            alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # For each alert, get its recipients
            for alert in alerts:
                cursor.execute('''
                SELECT * FROM alert_recipients WHERE alert_id = ?
                ''', (alert['alert_id'],))
                
                recipient_columns = [column[0] for column in cursor.description]
                recipients = [dict(zip(recipient_columns, row)) for row in cursor.fetchall()]
                alert['recipients'] = recipients
                
                # Parse JSON stored in additional_data
                if alert['additional_data']:
                    try:
                        alert['additional_data'] = json.loads(alert['additional_data'])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse additional_data for alert {alert['alert_id']}")
            
            logger.info(f"Retrieved {len(alerts)} historical alerts for user {user_id}")
            return alerts
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving alert history: {str(e)}")
            raise
    
    def get_alert_statistics(self, user_id=None, days=30):
        """
        Generate statistics about alerts.
        
        Args:
            user_id (str): Optional user ID to filter alerts
            days (int): Number of days to include in statistics
            
        Returns:
            dict: Alert statistics
        """
        try:
            cursor = self.db_connection.cursor()
            
            params = []
            where_clause = f"WHERE created_at >= datetime('now', '-{days} days')"
            
            if user_id:
                where_clause += " AND user_id = ?"
                params.append(user_id)
            
            # Get total alerts by severity
            cursor.execute(f'''
            SELECT severity, COUNT(*) as count
            FROM alerts
            {where_clause}
            GROUP BY severity
            ''', params)
            
            severity_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get response times (time between creation and acknowledgment)
            cursor.execute(f'''
            SELECT severity, 
                   AVG(JULIANDAY(acknowledged_at) - JULIANDAY(created_at)) * 24 * 60 as avg_response_minutes
            FROM alerts
            {where_clause}
            AND acknowledged = 1
            GROUP BY severity
            ''', params)
            
            response_times = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get source agent counts
            cursor.execute(f'''
            SELECT source_agent, COUNT(*) as count
            FROM alerts
            {where_clause}
            GROUP BY source_agent
            ''', params)
            
            source_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get resolution rate
            cursor.execute(f'''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved
            FROM alerts
            {where_clause}
            ''', params)
            
            resolution_data = cursor.fetchone()
            total_alerts = resolution_data[0] if resolution_data else 0
            resolved_alerts = resolution_data[1] if resolution_data else 0
            
            # Handle the case where there are no alerts to avoid division by zero
            resolution_rate = (resolved_alerts / total_alerts) if total_alerts > 0 else 0
            
            stats = {
                "period_days": days,
                "total_alerts": total_alerts,
                "severity_distribution": severity_counts,
                "average_response_times_minutes": response_times,
                "source_distribution": source_counts,
                "resolution_rate": resolution_rate
            }
            
            if user_id:
                stats["user_id"] = user_id
            
            logger.info(f"Generated alert statistics for the past {days} days")
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error generating alert statistics: {str(e)}")
            raise
    
    def create_alerts_from_coordinator_output(self, coordinator_output):
        """
        Create alerts from AgentCoordinator output.
        
        Args:
            coordinator_output (dict): Output from AgentCoordinator.process_all_data()
            
        Returns:
            list: IDs of created alerts
        """
        try:
            if not isinstance(coordinator_output, dict):
                logger.error(f"Invalid coordinator output type: {type(coordinator_output)}")
                return []
                
            if "actions" not in coordinator_output or not coordinator_output["actions"]:
                logger.info("No actions in coordinator output to create alerts from")
                return []
            
            user_id = coordinator_output.get("user_id", "unknown")
            alert_ids = []
            
            for action in coordinator_output["actions"]:
                # Map severity from coordinator to alert severity
                severity = action.get("severity", "info")
                
                # Map action type to source agent
                action_type = action.get("type", "")
                if "health" in action_type:
                    source_agent = "health_monitoring"
                elif "safety" in action_type:
                    source_agent = "safety_monitoring"
                elif "reminder" in action_type:
                    source_agent = "daily_reminder"
                else:
                    source_agent = "agent_coordinator"
                
                # Create the alert
                alert_id = self.create_alert(
                    user_id=user_id,
                    message=action.get("message", "No message provided"),
                    severity=severity,
                    source_agent=source_agent,
                    additional_data=action.get("data")
                )
                
                alert_ids.append(alert_id)
            
            logger.info(f"Created {len(alert_ids)} alerts from coordinator output")
            return alert_ids
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error creating alerts from coordinator output: {str(e)}\n{error_details}")
            return []
    
    def send_voice_alert(self, alert_text):
        """
        Send a voice alert using text-to-speech.
        
        Args:
            alert_text (str): Text to be spoken
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 1.0)
            tts_engine.say(alert_text)
            tts_engine.runAndWait()
            logger.info(f"Voice alert sent: {alert_text}")
            return True
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error sending voice alert: {str(e)}\n{error_details}")
            return False


if __name__ == "__main__":
    logger.info("Starting AlertSystem")

    try:
        # Initialize the AlertSystem
        alert_system = AlertSystem()
        logger.info("AlertSystem initialized successfully")
        
        # Test alert creation
        alert_id = alert_system.create_alert(
            user_id="test_user",
            message="This is a test alert",
            severity="info",
            source_agent="system_test"
        )
        logger.info(f"Created test alert with ID: {alert_id}")

        # Production-ready: Replace with API calls or scheduled tasks
        logger.info("AlertSystem is running. Waiting for tasks...")
        try:
            while True:
                # Example: Perform periodic maintenance or data processing
                time.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            logger.info("AlertSystem shutdown requested")

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in main function: {e}\n{error_details}")
    finally:
        if 'alert_system' in locals():
            alert_system.close_connection()
        logger.info("AlertSystem stopped")