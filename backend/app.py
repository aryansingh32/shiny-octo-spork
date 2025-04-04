import os
import sys
import argparse
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from integration.main import ElderlyCareSys
from integration.api import app as flask_app

# Conditionally import pyttsx3
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    print("Warning: pyttsx3 not available. Text-to-speech functionality will be disabled.")

# Load environment variables from .env file
load_dotenv()

# Setup logging with rotation
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize the ElderlyCareSys instance
try:
    system = ElderlyCareSys()
    
    # Import routes after system initialization to avoid circular imports
    from integration.api import register_routes
    register_routes(app, system)
except Exception as e:
    logger.error(f"Error initializing ElderlyCareSys: {e}")
    # Continue without the system for routes that don't require it

@app.route('/')
def home():
    return jsonify({
        "name": "ElderlyCareUI API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/api/docs",
        "tts_available": HAS_TTS
    })

@app.route('/api/docs')
def api_docs():
    return jsonify({
        "message": "API documentation",
        "endpoints": [
            {"path": "/api/health", "methods": ["GET"], "description": "Check API health"},
            {"path": "/api/users", "methods": ["GET"], "description": "Get all users"},
            {"path": "/api/alerts", "methods": ["GET"], "description": "Get active alerts"},
            {"path": "/api/reminders", "methods": ["GET", "POST"], "description": "Get all reminders or add a new one"}
        ],
        "github": "https://github.com/yourusername/ElderlyCareUI"
    })

def run_api(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask API server with proper error handling."""
    try:
        logger.info(f"Starting Elderly Care System API on {host}:{port}")
        
        # Print registered routes for debugging purposes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"Rule: {rule}, Methods: {rule.methods}")
        
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        raise

def run_cli():
    """Run the system in command-line interface mode with improved error handling."""
    system = None
    try:
        system = ElderlyCareSys()
        
        while True:
            try:
                print("\nElderly Care System CLI")
                print("1. Process data for a user")
                print("2. Get active alerts")
                print("3. Add a new user")
                print("4. Add a new reminder")
                print("5. Exit")
                
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    try:
                        user_id = int(input("Enter user ID: "))
                        results = system.process_data_for_user(user_id)
                        print(f"Results: {json.dumps(results, indent=2)}")
                    except ValueError:
                        print("Invalid user ID. Please enter a number.")
                    except Exception as e:
                        logger.error(f"Error processing data: {e}")
                        print(f"Error: {str(e)}")
                
                elif choice == '2':
                    try:
                        user_id_input = input("Enter user ID (or leave blank for all alerts): ").strip()
                        user_id = int(user_id_input) if user_id_input else None
                        alerts = system.alert_system.get_active_alerts(user_id)
                        print(f"Active alerts: {json.dumps(alerts, indent=2)}")
                    except ValueError:
                        print("Invalid user ID. Please enter a number or leave blank.")
                    except Exception as e:
                        logger.error(f"Error getting alerts: {e}")
                        print(f"Error: {str(e)}")
                
                elif choice == '3':
                    try:
                        name = input("Enter user name: ").strip()
                        if not name:
                            print("Name cannot be empty.")
                            continue
                            
                        age = int(input("Enter user age: "))
                        if age < 0 or age > 120:
                            print("Invalid age. Please enter a number between 0 and 120.")
                            continue
                            
                        preferences = input("Enter user preferences as JSON (or leave blank): ").strip()
                        preferences_dict = json.loads(preferences) if preferences else {}
                        
                        user_id = system.db.add_user(name, age, preferences_dict)
                        print(f"User added successfully with ID: {user_id}")
                    except ValueError as e:
                        print(f"Invalid input: {str(e)}")
                    except json.JSONDecodeError:
                        print("Invalid JSON format for preferences.")
                    except Exception as e:
                        logger.error(f"Error adding user: {e}")
                        print(f"Error: {str(e)}")
                
                elif choice == '4':
                    try:
                        user_id = int(input("Enter user ID: "))
                        title = input("Enter reminder title: ").strip()
                        if not title:
                            print("Title cannot be empty.")
                            continue
                            
                        reminder_time = input("Enter reminder time (YYYY-MM-DD HH:MM): ").strip()
                        try:
                            reminder_datetime = datetime.strptime(reminder_time, "%Y-%m-%d %H:%M")
                        except ValueError:
                            print("Invalid time format. Please use YYYY-MM-DD HH:MM.")
                            continue
                            
                        reminder_type = input("Enter reminder type (medication/appointment/activity): ").strip().lower()
                        if reminder_type not in ['medication', 'appointment', 'activity']:
                            print("Invalid reminder type. Please choose from: medication, appointment, activity.")
                            continue
                            
                        reminder_id = system.db.add_reminder(
                            user_id=user_id,
                            title=title,
                            reminder_type=reminder_type,
                            scheduled_time=reminder_datetime
                        )
                        print(f"Reminder added successfully with ID: {reminder_id}")
                    except ValueError as e:
                        print(f"Invalid input: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error adding reminder: {e}")
                        print(f"Error: {str(e)}")
                
                elif choice == '5':
                    print("Exiting Elderly Care System CLI.")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
            except Exception as e:
                logger.error(f"Unexpected error in CLI: {e}")
                print(f"An unexpected error occurred: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error initializing CLI: {e}")
        print(f"Error: {str(e)}")
    finally:
        if system:
            system.close()

def run_demo():
    """Run the demonstration script with error handling."""
    try:
        from demo.demo import run_demo
        run_demo()
    except ImportError:
        logger.error("Demo module not found")
        print("Demo module not found. Please ensure the demo package is installed.")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        print(f"Error running demo: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elderly Care Multi-Agent AI System")
    parser.add_argument('--mode', choices=['api', 'cli', 'demo'], default='api',
                      help='Run mode: api (REST API server), cli (command line), or demo')
    parser.add_argument('--host', default='0.0.0.0', help='Host for API server')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port for API server')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    try:
        args = parser.parse_args()
        
        if args.mode == 'api':
            if os.environ.get('FLASK_ENV') == 'production':
                # In production, we'll use gunicorn, this is just for local testing
                app.run(host=args.host, port=args.port, debug=False)
            else:
                run_api(host=args.host, port=args.port, debug=args.debug)
        elif args.mode == 'cli':
            run_cli()
        elif args.mode == 'demo':
            run_demo()
    except KeyboardInterrupt:
        logger.info("Application shutdown requested")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)