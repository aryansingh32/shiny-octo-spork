import logging
from typing import List, Dict, Any
from datetime import datetime

class AlertSystem:
    """
    A system for managing and processing alerts from various monitoring agents.
    """
    
    def __init__(self, database):
        """
        Initialize the AlertSystem with a database connection.
        
        Args:
            database: Database connection object
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
        
    def create_alert(self, alert_type: str, severity: str, message: str, user_id: int = None) -> Dict[str, Any]:
        """
        Create a new alert in the system.
        
        Args:
            alert_type: Type of alert (e.g., 'health', 'safety', 'reminder')
            severity: Severity level ('low', 'medium', 'high', 'critical')
            message: Alert message
            user_id: Optional user ID associated with the alert
            
        Returns:
            Dict containing the created alert information
        """
        try:
            alert = {
                'type': alert_type,
                'severity': severity,
                'message': message,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Store alert in database
            cursor = self.database.cursor()
            cursor.execute("""
                INSERT INTO alerts (type, severity, message, user_id, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert['type'],
                alert['severity'],
                alert['message'],
                alert['user_id'],
                alert['timestamp'],
                alert['status']
            ))
            self.database.commit()
            
            alert['id'] = cursor.lastrowid
            self.logger.info(f"Created new alert: {alert}")
            return alert
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")
            raise
            
    def get_alerts(self, user_id: int = None, status: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve alerts from the system.
        
        Args:
            user_id: Optional user ID to filter alerts
            status: Optional status to filter alerts ('active', 'resolved', etc.)
            
        Returns:
            List of alert dictionaries
        """
        try:
            query = "SELECT * FROM alerts"
            params = []
            
            conditions = []
            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(user_id)
            if status is not None:
                conditions.append("status = ?")
                params.append(status)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            cursor = self.database.cursor()
            cursor.execute(query, params)
            alerts = cursor.fetchall()
            
            return [dict(alert) for alert in alerts]
            
        except Exception as e:
            self.logger.error(f"Error retrieving alerts: {str(e)}")
            raise
            
    def update_alert_status(self, alert_id: int, new_status: str) -> bool:
        """
        Update the status of an alert.
        
        Args:
            alert_id: ID of the alert to update
            new_status: New status for the alert
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            cursor = self.database.cursor()
            cursor.execute("""
                UPDATE alerts
                SET status = ?
                WHERE id = ?
            """, (new_status, alert_id))
            
            self.database.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"Error updating alert status: {str(e)}")
            raise
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active alerts in the system.
        
        Returns:
            List of active alert dictionaries
        """
        return self.get_alerts(status='active')
        
    def get_user_alerts(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all alerts for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of alert dictionaries for the user
        """
        return self.get_alerts(user_id=user_id)
        
    def resolve_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if resolution was successful, False otherwise
        """
        return self.update_alert_status(alert_id, 'resolved') 