from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Load environment variables from the correct .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend.env')
load_dotenv(env_path)

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with more permissive settings for development
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080", "http://localhost:5173", "http://localhost:5000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Flask configuration
app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "False").lower() == "true"
app.config["ENV"] = os.getenv("FLASK_ENV", "production")

# Setup logging with rotation
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, "api.log"),
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ElderlyCareAPI")

# Import system components
from integration.agent_coordinator import AgentCoordinator
from integration.alert_system import AlertSystem
from integration.database import Database
from integration.main import ElderlyCareSys

# Initialize the system with error handling
try:
    system = ElderlyCareSys()
    logger.info("System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize system: {e}")
    raise

# Error handler for API
@app.errorhandler(Exception)
def handle_error(error):
    error_details = traceback.format_exc()
    logger.error(f"Unhandled exception: {error}\n{error_details}")
    response = {
        "success": False,
        "message": "An internal server error occurred. Please try again later."
    }
    return jsonify(response), 500

def validate_required_fields(data: Dict[str, Any], required_fields: list) -> tuple[bool, str]:
    """Validate that all required fields are present in the data."""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, ""

def validate_user_id(user_id: int) -> tuple[bool, str]:
    """Validate that the user ID exists."""
    try:
        if not system.database.get_user(user_id):
            return False, "User not found"
        return True, ""
    except Exception as e:
        return False, str(e)

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - provide basic info about the API"""
    return jsonify({
        "name": "Elderly Care System API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "health_check": "/api/health",
            "users": "/api/users",
            "alerts": "/api/alerts",
            "documentation": "/api/docs"
        }
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "Elderly Care System API",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/health/anomalies', methods=['GET'])
def get_health_anomalies():
    """Get health anomalies for all users"""
    try:
        anomalies = system.agent_coordinator.health_monitoring_agent.get_anomalies()
        return jsonify({"success": True, "anomalies": anomalies}), 200
    except Exception as e:
        logger.error(f"Error getting health anomalies: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/safety/alerts', methods=['GET'])
def get_safety_alerts():
    """Get safety alerts for all users"""
    try:
        alerts = system.agent_coordinator.safety_monitoring_agent.get_alerts()
        return jsonify({"success": True, "alerts": alerts}), 200
    except Exception as e:
        logger.error(f"Error getting safety alerts: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users', methods=['POST'])
def add_user():
    """Add a new user to the system"""
    data = request.json
    required_fields = ["name", "age"]
    
    is_valid, message = validate_required_fields(data, required_fields)
    if not is_valid:
        return jsonify({"success": False, "message": message}), 400
    
    try:
        # Validate age
        age = int(data["age"])
        if age < 0 or age > 120:
            return jsonify({"success": False, "message": "Age must be between 0 and 120"}), 400
            
        user_id = system.database.add_user(
            data["name"],
            age,
            data.get("preferences", {})
        )
        return jsonify({"success": True, "user_id": user_id}), 201
    except ValueError:
        return jsonify({"success": False, "message": "Invalid age format"}), 400
    except Exception as e:
        logger.error(f"Error adding user: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user information"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        user = system.database.get_user(user_id)
        return jsonify({"success": True, "user": user}), 200
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user information"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = request.json
        success = system.database.update_user(user_id, data)
        if success:
            return jsonify({"success": True, "message": "User updated successfully"}), 200
        return jsonify({"success": False, "message": "Failed to update user"}), 400
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user from the system"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        success = system.database.delete_user(user_id)
        if success:
            return jsonify({"success": True, "message": "User deleted successfully"}), 200
        return jsonify({"success": False, "message": "Failed to delete user"}), 400
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Get all users"""
    try:
        users = system.database.get_all_users()
        return jsonify({"success": True, "users": users}), 200
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# @app.route('/api/reminders', methods=['GET'])
# def get_agent_reminders():
#     """Get all reminders"""
#     try:
#         reminders = system.agent_coordinator.daily_reminder_agent.get_reminders()
#         return jsonify({"success": True, "reminders": reminders}), 200
#     except Exception as e:
#         logger.error(f"Error getting reminders: {e}")
#         return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_active_alerts():
    """Get active alerts, optionally filtered by user_id"""
    try:
        user_id = request.args.get('user_id', type=int)
        alerts = system.alert_system.get_active_alerts(user_id)
        return jsonify({"success": True, "alerts": alerts}), 200
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/alerts/count', methods=['GET'])
def get_active_alerts_count():
    """Get the count of active alerts"""
    try:
        count = system.alert_system.get_active_alerts_count()
        return jsonify({"success": True, "count": count}), 200
    except Exception as e:
        logger.error(f"Error getting alert count: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/reminders/count', methods=['GET'])
def get_reminders_count():
    """Get the count of reminders"""
    try:
        counts = system.database.get_reminders_count()
        # Add direct properties to make them accessible in the frontend
        return jsonify({
            "success": True, 
            "count": counts,
            "total": counts["total"],
            "upcoming": counts["upcoming"], 
            "missed": counts["missed"], 
            "completed": counts["completed"]
        }), 200
    except Exception as e:
        logger.error(f"Error getting reminder count: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health/abnormal-users', methods=['GET'])
def get_users_with_abnormal_readings():
    """Get users with abnormal health readings"""
    try:
        users = system.agent_coordinator.health_monitoring_agent.get_users_with_abnormal_readings()
        return jsonify({"success": True, "users": users}), 200
    except Exception as e:
        logger.error(f"Error getting abnormal users: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/alerts/attention', methods=['GET'])
def get_alerts_requiring_attention():
    """Get alerts requiring attention, filtered by severity"""
    try:
        severity = request.args.get('severity', type=str)
        alerts = system.alert_system.get_alerts_by_severity(severity)
        return jsonify({"success": True, "alerts": alerts}), 200
    except Exception as e:
        logger.error(f"Error getting attention alerts: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/reminders/upcoming', methods=['GET'])
def get_upcoming_reminders():
    """Get upcoming reminders"""
    try:
        reminders = system.agent_coordinator.daily_reminder_agent.get_upcoming_reminders()
        return jsonify({"success": True, "reminders": reminders}), 200
    except Exception as e:
        logger.error(f"Error getting upcoming reminders: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health/overview', methods=['GET'])
def get_health_overview():
    """Get health overview for all users"""
    try:
        overview = system.agent_coordinator.health_monitoring_agent.get_health_overview()
        return jsonify({"success": True, "overview": overview}), 200
    except Exception as e:
        logger.error(f"Error getting health overview: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health/report/<int:user_id>', methods=['GET'])
def get_health_report(user_id):
    """Get health report for a specific user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        report = system.agent_coordinator.health_monitoring_agent.get_health_report(user_id)
        return jsonify({"success": True, "report": report}), 200
    except Exception as e:
        logger.error(f"Error getting health report: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/count', methods=['GET'])
def get_total_users():
    """Get total number of users"""
    try:
        count = system.database.get_total_users()
        return jsonify({"success": True, "count": count}), 200
    except Exception as e:
        logger.error(f"Error getting user count: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/health-data', methods=['POST'])
def submit_health_data(user_id):
    """Submit health data for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = request.json
        system.database.store_health_data(user_id, data)
        results = system.process_health_data(user_id, data)
        return jsonify({
            "success": True,
            "message": "Health data processed successfully",
            "results": results
        }), 201
    except Exception as e:
        logger.error(f"Error submitting health data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/health-data/<string:metric>', methods=['GET'])
def get_health_data(user_id, metric):
    """Get specific health data for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = system.database.get_health_data(user_id, metric)
        return jsonify({"success": True, "data": data}), 200
    except Exception as e:
        logger.error(f"Error getting health data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/health-data/latest', methods=['GET'])
def get_latest_health_data(user_id):
    """Get latest health data for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = system.database.get_latest_health_data(user_id)
        return jsonify({"success": True, "data": data}), 200
    except Exception as e:
        logger.error(f"Error getting latest health data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/safety-data', methods=['POST'])
def submit_safety_data(user_id):
    """Submit safety data for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = request.json
        system.database.store_safety_data(user_id, data)
        results = system.process_safety_data(user_id, data)
        return jsonify({
            "success": True,
            "message": "Safety data processed successfully",
            "results": results
        }), 201
    except Exception as e:
        logger.error(f"Error submitting safety data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/reminders', methods=['POST'])
def add_reminder(user_id):
    """Add a reminder for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = request.json
        required_fields = ["title", "time", "type"]
        
        is_valid, message = validate_required_fields(data, required_fields)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 400
            
        # Validate reminder type
        if data["type"] not in ["medication", "appointment", "activity"]:
            return jsonify({"success": False, "message": "Invalid reminder type"}), 400
            
        # Validate time format
        try:
            reminder_time = datetime.strptime(data["time"], "%Y-%m-%d %H:%M")
        except ValueError:
            return jsonify({"success": False, "message": "Invalid time format"}), 400
            
        reminder_id = system.database.add_reminder(
            user_id=user_id,
            title=data["title"],
            reminder_type=data["type"],
            scheduled_time=data["time"],
            description=data.get("description"),
            priority=data.get("priority", 1),
            recurrence=data.get("recurrence")
        )
        return jsonify({
            "success": True,
            "message": "Reminder added successfully",
            "reminder_id": reminder_id
        }), 201
    except Exception as e:
        logger.error(f"Error adding reminder: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/reminders', methods=['GET'])
def get_reminders(user_id):
    """Get reminders for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        reminders = system.database.get_reminders(user_id)
        return jsonify({"success": True, "reminders": reminders}), 200
    except Exception as e:
        logger.error(f"Error getting reminders: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/reminders/<int:reminder_id>', methods=['PUT'])
def update_reminder(user_id, reminder_id):
    """Update a reminder for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            logger.error(f"Invalid user_id {user_id}: {message}")
            return jsonify({"success": False, "message": message}), 404
            
        data = request.json
        logger.info(f"Updating reminder {reminder_id} for user {user_id} with data: {data}")
        
        updated_reminder = system.database.update_reminder(reminder_id, user_id, data)
        
        if not updated_reminder:
            logger.error(f"Failed to update reminder {reminder_id} for user {user_id}")
            return jsonify({
                "success": False, 
                "message": f"Failed to update reminder {reminder_id}"
            }), 400
            
        logger.info(f"Successfully updated reminder {reminder_id} for user {user_id}")
        return jsonify({
            "success": True, 
            "message": "Reminder updated successfully",
            "reminder": updated_reminder
        }), 200
    except Exception as e:
        logger.error(f"Error updating reminder: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/reminders/<int:reminder_id>/acknowledge', methods=['POST'])
def acknowledge_reminder(user_id, reminder_id):
    """Acknowledge a reminder for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        # Update is_acknowledged and record the time
        data = {
            "is_acknowledged": 1,
            "acknowledged_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        updated_reminder = system.database.update_reminder(reminder_id, user_id, data)
        
        if not updated_reminder:
            return jsonify({
                "success": False, 
                "message": f"Failed to acknowledge reminder {reminder_id}"
            }), 400
            
        return jsonify({
            "success": True, 
            "message": "Reminder acknowledged successfully",
            "reminder": updated_reminder
        }), 200
    except Exception as e:
        logger.error(f"Error acknowledging reminder: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        success = system.alert_system.acknowledge_alert(alert_id)
        if success:
            return jsonify({"success": True, "message": "Alert acknowledged successfully"}), 200
        return jsonify({"success": False, "message": "Alert not found"}), 404
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/process-all/<int:user_id>', methods=['POST'])
def process_all_data(user_id):
    """Process all data for a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        results = system.process_data_for_user(user_id)
        return jsonify({
            "success": True,
            "message": "All data processed successfully",
            "results": results
        }), 200
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/dashboard/<int:user_id>', methods=['GET'])
def get_user_dashboard(user_id):
    """Get dashboard data for a specific user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        # Return mock dashboard data
        dashboard_data = {
            "user": system.database.get_user(user_id),
            "statistics": {
                "health": {
                    "normal_readings": 15,
                    "abnormal_readings": 2,
                    "trend": "stable"
                },
                "safety": {
                    "fall_risk": "low",
                    "recent_incidents": 0,
                    "trend": "good"
                },
                "reminders": {
                    "total": 8,
                    "completed": 6,
                    "missed": 0,
                    "upcoming": 2
                }
            },
            "recent_alerts": system.alert_system.get_active_alerts(user_id, limit=5),
            "upcoming_reminders": system.agent_coordinator.daily_reminder_agent.get_upcoming_reminders(days=3)
        }
        
        return jsonify({"success": True, "data": dashboard_data}), 200
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/stakeholders', methods=['POST'])
def add_stakeholder():
    """Add a new stakeholder"""
    data = request.json
    required_fields = ["name", "role", "contact"]
    
    is_valid, message = validate_required_fields(data, required_fields)
    if not is_valid:
        return jsonify({"success": False, "message": message}), 400
        
    try:
        stakeholder_id = system.database.add_stakeholder(data)
        return jsonify({
            "success": True,
            "message": "Stakeholder added successfully",
            "stakeholder_id": stakeholder_id
        }), 201
    except Exception as e:
        logger.error(f"Error adding stakeholder: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/users/<int:user_id>/stakeholders', methods=['POST'])
def link_stakeholder(user_id):
    """Link a stakeholder to a user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        data = request.json
        if "stakeholder_id" not in data:
            return jsonify({"success": False, "message": "Missing stakeholder_id"}), 400
            
        success = system.database.link_stakeholder_to_user(
            user_id,
            data["stakeholder_id"],
            data.get("alert_preferences", {})
        )
        if success:
            return jsonify({"success": True, "message": "Stakeholder linked successfully"}), 200
        return jsonify({"success": False, "message": "User or stakeholder not found"}), 404
    except Exception as e:
        logger.error(f"Error linking stakeholder: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/reminders/speak/<int:reminder_id>', methods=['POST'])
def speak_reminder(reminder_id):
    """
    Endpoint to speak a reminder using text-to-speech.
    
    Args:
        reminder_id (int): ID of the reminder to speak
        
    Returns:
        JSON response with status and message
    """
    try:
        # Import the DailyReminderAgent
        from daily_reminder.daily_reminder_agent import DailyReminderAgent
        # Get the database connection
        from integration.database import Database
        
        # Initialize the database connection
        db = Database()
        
        # Create a reminder agent
        reminder_agent = DailyReminderAgent(db.conn)
        
        # Speak the reminder
        success = reminder_agent.speak_reminder_by_id(reminder_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Successfully speaking reminder {reminder_id}'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to speak reminder {reminder_id}'
            }), 400
            
    except Exception as e:
        logger.error(f"Error speaking reminder {reminder_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/speak', methods=['POST'])
def speak_custom_message():
    """
    Endpoint to speak a custom message using text-to-speech.
    
    Request body:
    {
        "message": "Text to speak",
        "user_id": optional user ID,
        "priority": optional priority (high, medium, low),
        "voice": optional voice settings
    }
    
    Returns:
        JSON response with status and message
    """
    try:
        # Get the request data
        data = request.get_json()
        
        # Check for required fields
        if not data or 'message' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: message'
            }), 400
            
        # Import the voice reminder service
        from voice_reminder_service import get_voice_reminder_service
        
        # Get voice settings
        settings = data.get('voice_settings', {})
        
        # Set priority-specific settings
        priority = data.get('priority', 'medium')
        if priority == 'high':
            settings.update({
                'rate': 140,  # Slightly slower for emphasis
                'volume': 1.0,  # Full volume
            })
        elif priority == 'medium':
            settings.update({
                'rate': 150,  # Normal rate
                'volume': 0.9,  # Slightly lower volume
            })
        elif priority == 'low':
            settings.update({
                'rate': 160,  # Slightly faster
                'volume': 0.8,  # Lower volume
            })
            
        # Get the voice service
        voice_service = get_voice_reminder_service()
        
        # Queue the message
        success = voice_service.queue_reminder(
            text=data['message'],
            user_id=data.get('user_id'),
            settings=settings
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Successfully queued speech message'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to queue speech message'
            }), 400
            
    except Exception as e:
        logger.error(f"Error speaking custom message: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/voices', methods=['GET'])
def get_available_voices():
    """
    Endpoint to get available voices for text-to-speech.
    
    Returns:
        JSON response with list of available voices
    """
    try:
        # Import the voice reminder service
        from voice_reminder_service import get_voice_reminder_service
        
        # Get the voice service
        voice_service = get_voice_reminder_service()
        
        # Get available voices
        voices = voice_service.get_available_voices()
        
        return jsonify({
            'status': 'success',
            'voices': voices
        }), 200
            
    except Exception as e:
        logger.error(f"Error getting available voices: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

# Handle OPTIONS requests for CORS preflight
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 200

# Add missing ML insights endpoint
@app.route('/api/ml-insights/<int:user_id>', methods=['GET'])
def get_ml_insights(user_id):
    """Get ML-based insights for a specific user"""
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        # Return sample insights array (not an object)
        insights = [
            {
                "id": "insight-1",
                "type": "anomaly",
                "title": "Unusual heart rate pattern detected",
                "description": "Regular spikes in heart rate between 10pm-11pm may indicate stress before bed",
                "confidence": 0.78,
                "timestamp": datetime.now().isoformat(),
                "source": "health",
                "severity": "medium",
                "actions": [
                    {
                        "id": "action-1",
                        "label": "Schedule follow-up",
                        "endpoint": "/api/reminders",
                        "method": "POST",
                        "payload": {
                            "title": "Discuss heart rate pattern with doctor",
                            "type": "appointment"
                        }
                    }
                ]
            },
            {
                "id": "insight-2",
                "type": "recommendation",
                "title": "Consider evening medication adjustments",
                "description": "Frequent missed evening medications suggest a schedule adjustment may help",
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat(),
                "source": "reminder",
                "severity": "low"
            }
        ]
        
        return jsonify({"success": True, "insights": insights}), 200
    except Exception as e:
        logger.error(f"Error getting ML insights: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Add endpoint for ML model status
@app.route('/api/ml-models/status', methods=['GET'])
def get_ml_model_status():
    """Get the status of ML models in the system"""
    try:
        # Return the status of ML models
        models_status = {
            "models": {
                "health": True,  # True means the ML model is active, False means using fallback
                "safety": True,
                "reminder": False
            }
        }
        
        return jsonify(models_status), 200
    except Exception as e:
        logger.error(f"Error getting ML model status: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Add endpoint for adding reminders via ML action
@app.route('/api/reminders', methods=['POST', 'GET'])
def add_reminder_from_action():
    """
    Add a reminder from ML insights action or generic endpoint.
    
    Request format:
    {
        "title": "Reminder title",
        "type": "appointment|medication|activity|other",
        "time": "YYYY-MM-DD HH:MM" (optional, defaults to now + 1 day),
        "user_id": 123 (optional, defaults to first user),
        "description": "Additional details" (optional),
        "priority": 1|2|3 (optional, defaults to 2 - medium)
    }
    """
    # Return all reminders for GET requests
    if request.method == 'GET':
        try:
            reminders = system.agent_coordinator.daily_reminder_agent.get_reminders()
            return jsonify({"success": True, "reminders": reminders}), 200
        except Exception as e:
            logger.error(f"Error getting reminders: {e}")
            return jsonify({"success": False, "message": str(e)}), 500
    
    # Process POST requests to add a new reminder
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "Missing request body"}), 400
            
        # Required fields
        if 'title' not in data:
            return jsonify({"success": False, "message": "Missing required field: title"}), 400
        
        # Get optional fields with defaults
        reminder_type = data.get('type', 'appointment')
        if reminder_type not in ['appointment', 'medication', 'activity', 'other']:
            reminder_type = 'appointment'  # Default to appointment if invalid type
            
        # Time defaults to tomorrow same time
        scheduled_time = data.get('time')
        if not scheduled_time:
            tomorrow = datetime.now() + timedelta(days=1)
            scheduled_time = tomorrow.strftime('%Y-%m-%d %H:%M')
        
        # Get user_id, default to first user if not provided
        user_id = data.get('user_id')
        if not user_id:
            # Get first user from database
            users = system.database.get_all_users()
            if not users:
                return jsonify({"success": False, "message": "No users found"}), 400
            user_id = users[0]['id']
        
        # Optional fields
        description = data.get('description', '')
        priority = data.get('priority', 2)  # Default to medium priority
        
        # Add reminder to database
        reminder_id = system.database.add_reminder(
            user_id=user_id,
            title=data['title'],
            reminder_type=reminder_type,
            scheduled_time=scheduled_time,
            description=description,
            priority=priority
        )
        
        return jsonify({
            "success": True,
            "message": "Reminder added successfully",
            "reminder_id": reminder_id
        }), 201
    except Exception as e:
        logger.error(f"Error adding reminder from action: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Add endpoint for triggering health agent check manually
@app.route('/api/users/<int:user_id>/trigger-health-agent', methods=['POST', 'OPTIONS'])
def trigger_health_agent(user_id):
    """
    Trigger a health agent check for a specific user.
    
    Args:
        user_id (int): User ID to check
        
    Returns:
        JSON with results of health check
    """
    # Handle OPTIONS requests for CORS
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        is_valid, message = validate_user_id(user_id)
        if not is_valid:
            return jsonify({"success": False, "message": message}), 404
            
        # Run health check with the agent
        health_agent = system.agent_coordinator.health_monitoring_agent
        
        # Check if health agent has the run_health_check method
        if hasattr(health_agent, 'run_health_check'):
            results = health_agent.run_health_check(user_ids=[user_id], use_ml=True)
        else:
            # Fallback to process_data_for_user
            results = system.process_data_for_user(user_id)
            
        return jsonify({
            "success": True,
            "message": "Health agent check triggered successfully",
            "results": results
        }), 200
    except Exception as e:
        logger.error(f"Error triggering health agent for user {user_id}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))