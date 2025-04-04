"""
Cloud fallback module for ElderlyCareSys.
This provides minimal implementations for deployment environments 
where certain packages like pyttsx3 are not available.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MockSystem:
    """A base class that logs actions instead of performing them"""
    def __init__(self, name):
        self.name = name
        logger.info(f"Initialized mock {name}")
    
    def __getattr__(self, name):
        def method(*args, **kwargs):
            logger.info(f"Called {self.name}.{name} with args={args}, kwargs={kwargs}")
            return {"status": "mocked", "method": name, "args": args, "kwargs": kwargs}
        return method

class CloudElderlyCareSys:
    """Cloud-friendly version of ElderlyCareSys with fallbacks for missing dependencies"""
    
    def __init__(self):
        logger.info("Initializing CloudElderlyCareSys")
        self.db = MockSystem("DatabaseSystem")
        self.alert_system = CloudAlertSystem()
        self.voice_reminder_service = MockSystem("VoiceReminderService")
        self.user_count = 3  # Mock user count
        
    def process_data_for_user(self, user_id):
        """Process data for a user"""
        logger.info(f"Processing data for user {user_id}")
        return {
            "status": "processed",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def start_autonomous_operations(self):
        """Start autonomous operations"""
        logger.info("Started autonomous operations in cloud mode")
        return True
    
    def close(self):
        """Clean up resources"""
        logger.info("Closing CloudElderlyCareSys")
        return True

class CloudAlertSystem:
    """Cloud-friendly version of the alert system"""
    
    def __init__(self):
        logger.info("Initializing CloudAlertSystem")
        self.active_alerts = [
            {
                "id": 1,
                "user_id": 1,
                "type": "health",
                "severity": "medium",
                "message": "Blood pressure slightly elevated",
                "timestamp": "2025-04-04T18:30:00",
                "status": "active"
            },
            {
                "id": 2,
                "user_id": 2,
                "type": "safety",
                "severity": "high",
                "message": "Potential fall detected",
                "timestamp": "2025-04-04T19:15:00",
                "status": "active"
            }
        ]
    
    def get_active_alerts(self, user_id=None):
        """Get active alerts for a user or all users"""
        if user_id:
            return [alert for alert in self.active_alerts if alert["user_id"] == user_id]
        return self.active_alerts
    
    def add_alert(self, user_id, alert_type, severity, message):
        """Add a new alert"""
        new_alert = {
            "id": len(self.active_alerts) + 1,
            "user_id": user_id,
            "type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }
        self.active_alerts.append(new_alert)
        return new_alert["id"] 