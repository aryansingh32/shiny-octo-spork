import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
import traceback
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Optional, Dict, List, Any, Union

# Load environment variables from .env file
load_dotenv()

# Set up logging with rotation
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'database.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Database")

class Database:
    def __init__(self, db_path: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Database connection with improved error handling and configuration.

        Args:
            db_path (str, optional): Path to the SQLite database file.
            timeout (int): Connection timeout in seconds.
        """
        self.db_path = db_path or os.getenv("DATABASE_PATH", "default_database.db")
        self.timeout = timeout
        self.conn = None
        self.cursor = None
        
        try:
            # Resolve absolute path
            self.db_path = os.path.abspath(self.db_path)
            logger.info(f"Initializing database connection to {self.db_path}")
            
            # Establish the database connection with timeout
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=self.timeout
            )
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            
            # Enable foreign key constraints
            self.cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create necessary tables
            self._create_tables()
            
            # Create indexes
            self._create_indexes()
            
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            self._cleanup()
            raise Exception(f"Failed to initialize database: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Safely close database connections."""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self.conn = None
        self.cursor = None

    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor operations."""
        cursor = self.conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _create_indexes(self):
        """Create indexes for frequently queried columns."""
        try:
            with self._get_cursor() as cursor:
                # Indexes for users table
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_name ON users(name)')
                self.conn.commit()
                
                # Indexes for reminders table
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_reminders_user_id ON reminders(user_id)')
                self.conn.commit()
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_reminders_scheduled_time ON reminders(scheduled_time)')
                self.conn.commit()
                
                # Indexes for alerts table
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id)')
                self.conn.commit()
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
                self.conn.commit()
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)')
                self.conn.commit()
                
                # Indexes for health data table
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_data_user_id ON health_data(user_id)')
                self.conn.commit()
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_data_timestamp ON health_data(timestamp)')
                self.conn.commit()
                
                # Indexes for safety data table
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_safety_data_user_id ON safety_data(user_id)')
                self.conn.commit()
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_safety_data_timestamp ON safety_data(timestamp)')
                self.conn.commit()
                
                logger.info("Database indexes created successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Error creating indexes: {e}")
            self.conn.rollback()
            raise

    def _validate_json(self, data: Any, field_name: str) -> Optional[str]:
        """Validate and convert data to JSON string."""
        if data is None:
            return None
        try:
            if isinstance(data, str):
                # Try to parse the string to validate it's valid JSON
                json.loads(data)
                return data
            return json.dumps(data)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON data for {field_name}")
            raise ValueError(f"Invalid JSON data for {field_name}")

    def _validate_timestamp(self, timestamp: str) -> str:
        """Validate timestamp format."""
        try:
            datetime.fromisoformat(timestamp)
            return timestamp
        except ValueError:
            logger.error(f"Invalid timestamp format: {timestamp}")
            raise ValueError(f"Invalid timestamp format: {timestamp}")

    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        Retrieve user details from the database with improved error handling.

        Args:
            user_id (int): The ID of the user to retrieve.

        Returns:
            dict: A dictionary containing user details, or None if the user does not exist.
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                user = cursor.fetchone()
                
                if user:
                    user_dict = dict(user)
                    # Parse JSON fields
                    for field in ['preferences', 'emergency_contacts', 'medical_conditions']:
                        if user_dict.get(field):
                            try:
                                user_dict[field] = json.loads(user_dict[field])
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in {field} for user {user_id}")
                                user_dict[field] = None
                    return user_dict
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving user {user_id}: {e}")
            raise

    def _create_tables(self):
        """Create all necessary database tables if they don't exist."""
        try:
            cursor = self.cursor
            
            # Users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                emergency_contacts TEXT,
                medical_conditions TEXT
            )
            ''')
            
            # Caregivers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS caregivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                contact_info TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # User-Caregiver relationship table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_caregivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                caregiver_id INTEGER NOT NULL,
                relationship TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (caregiver_id) REFERENCES caregivers (id)
            )
            ''')
            
            # Health data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                heart_rate REAL,
                blood_pressure TEXT,
                temperature REAL,
                oxygen_level REAL,
                glucose_level REAL,
                additional_metrics TEXT,
                is_abnormal BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Safety data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS safety_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location TEXT,
                movement_type TEXT,
                fall_detected BOOLEAN,
                activity_level REAL,
                time_inactive INTEGER,
                risk_level TEXT,
                additional_data TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Reminders table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                reminder_type TEXT NOT NULL,
                scheduled_time TIMESTAMP NOT NULL,
                priority INTEGER DEFAULT 1,
                recurrence TEXT,
                is_acknowledged BOOLEAN DEFAULT 0,
                acknowledged_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Alerts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_acknowledged BOOLEAN DEFAULT 0,
                acknowledged_time TIMESTAMP,
                acknowledged_by TEXT,
                additional_data TEXT,
                resolved BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Alert recipients table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_recipients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER NOT NULL,
                recipient_id INTEGER NOT NULL,
                recipient_type TEXT NOT NULL,
                notification_sent BOOLEAN DEFAULT 0,
                notification_time TIMESTAMP,
                FOREIGN KEY (alert_id) REFERENCES alerts (id)
            )
            ''')
            
            # System logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                log_type TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                additional_data TEXT
            )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise Exception(f"Error creating tables: {e}")

    def get_all_users(self):
        """
        Retrieve all users from the database.
        
        Returns:
            list: A list of dictionaries containing user information.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY id")
            
            users = []
            for row in cursor.fetchall():
                user_dict = dict(row)
                users.append(user_dict)
                
            return users
        except Exception as e:
            logger.error(f"Error retrieving all users: {e}")
            return []

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield
            self.conn.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise

    def add_user(self, name: str, age: int, preferences: Optional[Dict] = None,
                emergency_contacts: Optional[List] = None,
                medical_conditions: Optional[List] = None) -> int:
        """
        Add a new user to the database with improved validation and error handling.

        Args:
            name (str): User name.
            age (int): User age.
            preferences (dict, optional): User preferences.
            emergency_contacts (list, optional): Emergency contacts.
            medical_conditions (list, optional): Medical conditions.

        Returns:
            int: The ID of the newly added user.

        Raises:
            ValueError: If input validation fails.
            sqlite3.Error: If database operation fails.
        """
        try:
            # Input validation
            if not name or not isinstance(name, str):
                raise ValueError("Name must be a non-empty string")
            if not isinstance(age, int) or age < 0 or age > 120:
                raise ValueError("Age must be a positive integer between 0 and 120")

            # Validate and convert JSON data
            preferences_json = self._validate_json(preferences, "preferences")
            emergency_contacts_json = self._validate_json(emergency_contacts, "emergency_contacts")
            medical_conditions_json = self._validate_json(medical_conditions, "medical_conditions")

            with self.transaction():
                with self._get_cursor() as cursor:
                    cursor.execute('''
            INSERT INTO users (name, age, preferences, emergency_contacts, medical_conditions)
            VALUES (?, ?, ?, ?, ?)
                    ''', (name, age, preferences_json, emergency_contacts_json, medical_conditions_json))
                    
                    user_id = cursor.lastrowid
                    logger.info(f"Added new user with ID {user_id}")
                    return user_id

        except sqlite3.Error as e:
            logger.error(f"Database error adding user: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding user: {e}")
            raise

    def add_users_batch(self, users: List[Dict]) -> List[int]:
        """
        Add multiple users in a batch operation.

        Args:
            users (List[Dict]): List of user dictionaries with name, age, and optional fields.

        Returns:
            List[int]: List of new user IDs.
            
        Raises:
            ValueError: If validation fails for any user.
            sqlite3.Error: If database operation fails.
        """
        user_ids = []
        
        try:
            with self.transaction():
                for user in users:
                    user_id = self.add_user(
                        name=user['name'],
                        age=user['age'],
                        preferences=user.get('preferences'),
                        emergency_contacts=user.get('emergency_contacts'),
                        medical_conditions=user.get('medical_conditions')
                    )
                    user_ids.append(user_id)
            
            return user_ids
        except (ValueError, sqlite3.Error) as e:
            logger.error(f"Error adding users in batch: {e}")
            raise

    def update_user(self, user_id: int, data: Dict) -> bool:
        """
        Update user information in the database.
        
        Args:
            user_id (int): ID of the user to update
            data (Dict): Dictionary containing user data to update
                Possible keys: name, age, preferences, emergency_contacts, medical_conditions
                
        Returns:
            bool: True if update was successful, False otherwise
            
        Raises:
            ValueError: If validation fails
            sqlite3.Error: If database operation fails
        """
        try:
            # Validate inputs
            if not isinstance(user_id, int) or user_id <= 0:
                raise ValueError(f"Invalid user_id: {user_id}")
                
            # Check if user exists
            if not self.get_user(user_id):
                logger.error(f"User with ID {user_id} not found")
                return False
                
            # Build the SET clause and parameters for the SQL query
            update_fields = []
            params = []
            
            if 'name' in data:
                if not isinstance(data['name'], str) or not data['name'].strip():
                    raise ValueError("Name must be a non-empty string")
                update_fields.append("name = ?")
                params.append(data['name'])
                
            if 'age' in data:
                if not isinstance(data['age'], int) or data['age'] < 0 or data['age'] > 120:
                    raise ValueError("Age must be a number between 0 and 120")
                update_fields.append("age = ?")
                params.append(data['age'])
                
            if 'preferences' in data:
                json_prefs = self._validate_json(data['preferences'], 'preferences')
                update_fields.append("preferences = ?")
                params.append(json_prefs)
                
            if 'emergency_contacts' in data:
                json_contacts = self._validate_json(data['emergency_contacts'], 'emergency_contacts')
                update_fields.append("emergency_contacts = ?")
                params.append(json_contacts)
                
            if 'medical_conditions' in data:
                json_conditions = self._validate_json(data['medical_conditions'], 'medical_conditions')
                update_fields.append("medical_conditions = ?")
                params.append(json_conditions)
                
            # If no fields to update, return false
            if not update_fields:
                logger.warning("No valid fields to update")
                return False
                
            # Create and execute the SQL query
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            params.append(user_id)
            
            with self.transaction():
                cursor = self.conn.cursor()
                cursor.execute(query, params)
                
            # Check if update was successful
            return cursor.rowcount > 0
                
        except ValueError as e:
            logger.error(f"Validation error updating user {user_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error updating user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating user {user_id}: {e}")
            raise

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user from the database.
        
        Args:
            user_id (int): ID of the user to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
            
        Raises:
            ValueError: If validation fails
            sqlite3.Error: If database operation fails
        """
        try:
            # Validate inputs
            if not isinstance(user_id, int) or user_id <= 0:
                raise ValueError(f"Invalid user_id: {user_id}")
                
            # Check if user exists
            if not self.get_user(user_id):
                logger.error(f"User with ID {user_id} not found")
                return False
                
            # Delete the user
            with self.transaction():
                cursor = self.conn.cursor()
                
                # Delete associated data first to maintain referential integrity
                # This assumes foreign key constraints are enabled
                tables_with_user_id = [
                    "reminders", 
                    "alerts", 
                    "health_data", 
                    "safety_data", 
                    "user_caregivers"
                ]
                
                for table in tables_with_user_id:
                    cursor.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                    logger.info(f"Deleted {cursor.rowcount} records from {table} for user {user_id}")
                
                # Now delete the user
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                
                # Check if deletion was successful
                if cursor.rowcount > 0:
                    logger.info(f"Successfully deleted user {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID {user_id} to delete")
                    return False
                
        except ValueError as e:
            logger.error(f"Validation error deleting user {user_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error deleting user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting user {user_id}: {e}")
            raise

    def get_active_alerts(self, user_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """
        Get active alerts from the database.

        Args:
            user_id (int, optional): Filter alerts for a specific user.
            limit (int): Maximum number of alerts to return.

        Returns:
            list: List of dictionaries containing alert information.
        """
        try:
            with self._get_cursor() as cursor:
                query = '''
                SELECT a.*, u.name as user_name
                FROM alerts a
                JOIN users u ON a.user_id = u.id
                WHERE a.is_acknowledged = 0
                '''
                params = []

                if user_id is not None:
                    query += " AND a.user_id = ?"
                    params.append(user_id)

                query += " ORDER BY a.created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                alerts = []
                for row in cursor.fetchall():
                    alert = dict(row)
                    # Parse additional_data if present
                    if alert.get('additional_data'):
                        try:
                            alert['additional_data'] = json.loads(alert['additional_data'])
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in additional_data for alert {alert['id']}")
                            alert['additional_data'] = None
                    alerts.append(alert)

                return alerts

        except sqlite3.Error as e:
            logger.error(f"Database error retrieving active alerts: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving active alerts: {e}")
            raise

    def get_reminders_count(self):
        """Get count of reminders"""
        try:
            cursor = self.conn.cursor()
            
            # First check if the 'status' column exists
            cursor.execute("PRAGMA table_info(reminders)")
            columns = cursor.fetchall()
            column_names = [column['name'] for column in columns]
            has_status_column = 'status' in column_names
            
            logger.info(f"Reminders table columns: {column_names}")
            logger.info(f"Using status column: {has_status_column}")
            
            # Get total reminders count
            cursor.execute("SELECT COUNT(*) FROM reminders")
            total = cursor.fetchone()[0]
            
            # Get upcoming, missed, and completed counts based on the schema
            if has_status_column:
                cursor.execute("SELECT COUNT(*) FROM reminders WHERE status = 'pending'")
                upcoming = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM reminders WHERE status = 'missed'")
                missed = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM reminders WHERE status = 'completed'")
                completed = cursor.fetchone()[0]
            else:
                # Use is_acknowledged field instead
                # Upcoming: not acknowledged and scheduled in the future
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("SELECT COUNT(*) FROM reminders WHERE is_acknowledged = 0 AND scheduled_time > ?", (now,))
                upcoming = cursor.fetchone()[0]
                
                # Missed: not acknowledged and scheduled in the past
                cursor.execute("SELECT COUNT(*) FROM reminders WHERE is_acknowledged = 0 AND scheduled_time <= ?", (now,))
                missed = cursor.fetchone()[0]
                
                # Completed: acknowledged
                cursor.execute("SELECT COUNT(*) FROM reminders WHERE is_acknowledged = 1")
                completed = cursor.fetchone()[0]
            
            logger.info(f"Reminder counts: total={total}, upcoming={upcoming}, missed={missed}, completed={completed}")
            
            return {
                "total": total,
                "upcoming": upcoming,
                "missed": missed,
                "completed": completed
            }
        except Exception as e:
            logger.error(f"Error getting reminders count: {str(e)}")
            return {"total": 0, "upcoming": 0, "missed": 0, "completed": 0}
            
    def update_reminder(self, reminder_id, user_id, data):
        """Update a reminder's fields
        
        Args:
            reminder_id (int): ID of the reminder to update
            user_id (int): ID of the user who owns the reminder
            data (dict): Data fields to update
            
        Returns:
            dict: Updated reminder data
        """
        try:
            # Check if the 'status' column exists
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(reminders)")
            columns = cursor.fetchall()
            column_names = [column['name'] for column in columns]
            has_status_column = 'status' in column_names
            
            logger.info(f"Reminders table columns: {column_names}")
            logger.info(f"Has status column: {has_status_column}")
            
            # First check if the reminder exists
            query = "SELECT * FROM reminders WHERE id = ? AND user_id = ?"
            logger.info(f"Checking if reminder exists: {query} with values {(reminder_id, user_id)}")
            
            cursor.execute(query, (reminder_id, user_id))
            reminder = cursor.fetchone()
            
            if not reminder:
                logger.error(f"Reminder {reminder_id} not found for user {user_id}")
                # Let's query to see if the reminder exists at all, regardless of user
                cursor.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
                any_reminder = cursor.fetchone()
                if any_reminder:
                    user_in_reminder = dict(any_reminder).get('user_id')
                    logger.error(f"Reminder {reminder_id} exists but belongs to user {user_in_reminder}, not {user_id}")
                else:
                    logger.error(f"Reminder {reminder_id} does not exist in the database")
                return None
                
            # Build the update query based on the fields in data
            update_fields = []
            update_values = []
            
            # Map frontend field names to database column names based on available columns
            field_mapping = {
                "title": "title",
                "description": "description",
                "priority": "priority",
                "recurrence": "recurrence",
                "type": "reminder_type",
                "dateTime": "scheduled_time",
            }
            
            # Add status-related mappings based on column presence
            if has_status_column:
                field_mapping["completed"] = "status"  # Map 'completed: true' to 'status: completed'
                field_mapping["status"] = "status"
            else:
                field_mapping["completed"] = "is_acknowledged"  # Map 'completed: true' to 'is_acknowledged: 1'
                field_mapping["is_acknowledged"] = "is_acknowledged"
                field_mapping["acknowledged_time"] = "acknowledged_time"
            
            # Log the incoming data before processing
            logger.info(f"Incoming data for reminder update: {data}")
            logger.info(f"Field mapping being used: {field_mapping}")
            
            for key, value in data.items():
                if key in field_mapping:
                    # Special handling for completed field
                    if key == "completed" and value:
                        if has_status_column:
                            update_fields.append(f"{field_mapping[key]} = ?")
                            update_values.append("completed")
                        else:
                            update_fields.append(f"{field_mapping[key]} = ?")
                            update_values.append(1)  # Set is_acknowledged to 1
                            update_fields.append("acknowledged_time = ?")
                            update_values.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    else:
                        update_fields.append(f"{field_mapping[key]} = ?")
                        update_values.append(value)
            
            if not update_fields:
                logger.error("No valid fields to update")
                return None
                
            # Add reminder_id and user_id to values
            update_values.append(reminder_id)
            update_values.append(user_id)
            
            # Execute update
            query = f"""
                UPDATE reminders 
                SET {', '.join(update_fields)}
                WHERE id = ? AND user_id = ?
            """
            logger.info(f"Executing update query: {query} with values {update_values}")
            
            cursor.execute(query, update_values)
            self.conn.commit()

            # Get updated reminder
            query = """
                SELECT r.*, u.name as user_name 
                FROM reminders r 
                JOIN users u ON r.user_id = u.id 
                WHERE r.id = ?
            """
            logger.info(f"Fetching updated reminder: {query} with value {reminder_id}")
            
            cursor.execute(query, (reminder_id,))
            updated_reminder = cursor.fetchone()
            
            if updated_reminder:
                result = dict(updated_reminder)
                logger.info(f"Successfully updated reminder: {result}")
                return result
                
            logger.error(f"Failed to fetch updated reminder {reminder_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error updating reminder {reminder_id}: {str(e)}\n{traceback.format_exc()}")
            return None

    def get_total_users(self) -> int:
        """
        Get the total number of users in the database.

        Returns:
            int: Total number of users.
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM users")
                result = cursor.fetchone()
                return result['count'] or 0

        except sqlite3.Error as e:
            logger.error(f"Database error getting total users: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting total users: {e}")
            raise

    def close(self):
        """Safely close the database connection."""
        self._cleanup()
        logger.info("Database connection closed")

    def __del__(self):
        """Ensure database connection is closed when object is destroyed."""
        self.close()

    def add_reminder(self, user_id, title, reminder_type, scheduled_time, description=None, priority=1, recurrence=None):
        """
        Add a new reminder to the database.

        Args:
            user_id (int): The ID of the user the reminder is for.
            title (str): The title of the reminder.
            reminder_type (str): The type of the reminder (e.g., "system", "medication").
            scheduled_time (str): The scheduled time for the reminder (ISO format).
            description (str, optional): A description of the reminder.
            priority (int, optional): The priority of the reminder (default is 1).
            recurrence (str, optional): Recurrence pattern for the reminder.

        Returns:
            int: The ID of the newly added reminder.
        """
        try:
            # Insert the reminder into the database
            query = '''
            INSERT INTO reminders (user_id, title, description, reminder_type, scheduled_time, priority, recurrence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            self.cursor.execute(query, (user_id, title, description, reminder_type, scheduled_time, priority, recurrence))
            self.conn.commit()

            # Return the ID of the newly added reminder
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error adding reminder to the database: {e}")
            raise Exception(f"Error adding reminder to the database: {e}")

    def get_reminders(self, user_id=None, only_pending=True, start_date=None, end_date=None):
        """
        Get reminders for a specific user within a given date range.
        
        Args:
            user_id (int, optional): User ID to filter reminders
            only_pending (bool, optional): Only return pending reminders
            start_date (str, optional): ISO format start date
            end_date (str, optional): ISO format end date
            
        Returns:
            list: A list of reminders
        """
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM reminders WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
                
            if only_pending:
                query += " AND is_acknowledged = 0"
                
            if start_date:
                query += " AND scheduled_time >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND scheduled_time <= ?"
                params.append(end_date)
                
            query += " ORDER BY scheduled_time"
            
            cursor.execute(query, params)
            
            reminders = []
            for row in cursor.fetchall():
                reminder = dict(row)
                reminders.append(reminder)
                
            return reminders
        except sqlite3.Error as e:
            logger.error(f"Error retrieving reminders: {e}")
            return []

    def get_all_reminders(self):
        """
        Get all reminders from the database.
        
        Returns:
            list: A list of all reminders
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM reminders ORDER BY scheduled_time")
            
            reminders = []
            for row in cursor.fetchall():
                reminder = dict(row)
                reminders.append(reminder)
                
            return reminders
        except sqlite3.Error as e:
            logger.error(f"Error retrieving all reminders: {e}")
            return []

    def add_alert_recipient(self, alert_id, recipient_id, recipient_type):
        """
        Add a recipient to an alert.
        
        Args:
            alert_id (int): Alert ID
            recipient_id (int): Recipient ID
            recipient_type (str): Type of recipient (caregiver, family, etc.)
            
        Returns:
            int: ID of the newly added recipient
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO alert_recipients (alert_id, recipient_id, recipient_type) 
                VALUES (?, ?, ?)
            """, (alert_id, recipient_id, recipient_type))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error adding alert recipient: {e}")
            return None

    def log_system_event(self, log_type, component, message, additional_data=None):
        """
        Log a system event to the database.
        
        Args:
            log_type (str): Type of log
            component (str): Component generating the log
            message (str): Log message
            additional_data (dict, optional): Additional data
            
        Returns:
            int: ID of the newly added log entry
        """
        try:
            additional_data_json = json.dumps(additional_data) if additional_data else None
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO system_logs (log_type, component, message, additional_data)
                VALUES (?, ?, ?, ?)
            """, (log_type, component, message, additional_data_json))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error logging system event: {e}")
            return None

    def get_health_data(self, user_id, limit=100, start_date=None, end_date=None):
        """
        Get health data for a specific user.
        
        Args:
            user_id (int): User ID
            limit (int, optional): Maximum number of records
            start_date (str, optional): ISO format start date
            end_date (str, optional): ISO format end date
            
        Returns:
            list: A list of health data records
        """
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM health_data WHERE user_id = ?"
            params = [user_id]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            health_data = []
            for row in cursor.fetchall():
                data = dict(row)
                health_data.append(data)
                
            return health_data
        except sqlite3.Error as e:
            logger.error(f"Error retrieving health data: {e}")
            return []

    def get_health_stats(self, user_id, days=7):
        """
        Get health statistics for a user over a time period.
        
        Args:
            user_id (int): User ID
            days (int, optional): Number of days to include
            
        Returns:
            dict: Health statistics
        """
        try:
            # This is a placeholder - implement actual statistics calculation
            return {
                "average_heart_rate": 75,
                "average_blood_pressure": "120/80",
                "average_temperature": 98.6,
                "average_oxygen_level": 98,
                "abnormal_readings": 0
            }
        except Exception as e:
            logger.error(f"Error calculating health stats: {e}")
            return {}

    def get_safety_data(self, user_id, limit=100, start_date=None, end_date=None):
        """
        Get safety data for a specific user.
        
        Args:
            user_id (int): User ID
            limit (int, optional): Maximum number of records
            start_date (str, optional): ISO format start date
            end_date (str, optional): ISO format end date
            
        Returns:
            list: A list of safety data records
        """
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM safety_data WHERE user_id = ?"
            params = [user_id]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            safety_data = []
            for row in cursor.fetchall():
                data = dict(row)
                safety_data.append(data)
                
            return safety_data
        except sqlite3.Error as e:
            logger.error(f"Error retrieving safety data: {e}")
            return []

    def get_safety_stats(self, user_id, days=7):
        """
        Get safety statistics for a user over a time period.
        
        Args:
            user_id (int): User ID
            days (int, optional): Number of days to include
            
        Returns:
            dict: Safety statistics
        """
        try:
            # This is a placeholder - implement actual statistics calculation
            return {
                "falls_detected": 0,
                "average_activity_level": 65,
                "longest_inactive_period": 480,  # minutes
                "high_risk_incidents": 0
            }
        except Exception as e:
            logger.error(f"Error calculating safety stats: {e}")
            return {}

    def get_reminder_stats(self, user_id, days=7):
        """
        Get reminder statistics for a user over a time period.
        
        Args:
            user_id (int): User ID
            days (int, optional): Number of days to include
            
        Returns:
            dict: Reminder statistics
        """
        try:
            # This is a placeholder - implement actual statistics calculation
            return {
                "total_reminders": 15,
                "acknowledged": 12,
                "missed": 3,
                "acknowledgement_rate": 80.0  # percentage
            }
        except Exception as e:
            logger.error(f"Error calculating reminder stats: {e}")
            return {}

    def get_latest_health_data(self, user_id: int) -> Dict:
        """
        Get the latest health data for a specific user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Latest health data record or empty dict if no data exists
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM health_data 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return {}
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving latest health data: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error retrieving latest health data: {e}")
            return {}

    def get_latest_safety_data(self, user_id: int) -> Dict:
        """
        Get the latest safety data for a specific user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Latest safety data record or empty dict if no data exists
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM safety_data 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return {}
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving latest safety data: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error retrieving latest safety data: {e}")
            return {}

    def store_health_data(self, user_id: int, data: dict) -> bool:
        """
        Store health data for a user in the database.
        
        Args:
            user_id (int): ID of the user
            data (dict): Health data to store
                Keys can include: heart_rate, blood_pressure, temperature, oxygen_level,
                glucose_level, and any additional_metrics as JSON
                
        Returns:
            bool: True if data was stored successfully, False otherwise
            
        Raises:
            ValueError: If validation fails
            sqlite3.Error: If database operation fails
        """
        try:
            # Validate inputs
            if not isinstance(user_id, int) or user_id <= 0:
                raise ValueError(f"Invalid user_id: {user_id}")
                
            # Check if user exists
            if not self.get_user(user_id):
                logger.error(f"User with ID {user_id} not found")
                return False
            
            # Extract and validate health metrics
            heart_rate = data.get('heart_rate')
            blood_pressure = data.get('blood_pressure')
            temperature = data.get('temperature')
            oxygen_level = data.get('oxygen_level')
            glucose_level = data.get('glucose_level')
            
            # Additional metrics as JSON
            additional_metrics = data.get('additional_metrics', {})
            additional_metrics_json = json.dumps(additional_metrics) if additional_metrics else None
            
            # Auto-detect abnormal readings
            is_abnormal = False
            
            # Basic abnormality checks (these can be improved with more medical knowledge)
            if heart_rate and (heart_rate < 40 or heart_rate > 180):
                is_abnormal = True
                
            if temperature and (temperature < 35 or temperature > 39):
                is_abnormal = True
                
            if oxygen_level and oxygen_level < 92:
                is_abnormal = True
            
            # Insert the health data
            with self.transaction():
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO health_data (
                    user_id, heart_rate, blood_pressure, temperature, 
                    oxygen_level, glucose_level, additional_metrics, is_abnormal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, heart_rate, blood_pressure, temperature,
                    oxygen_level, glucose_level, additional_metrics_json, is_abnormal
                ))
                
                data_id = cursor.lastrowid
                logger.info(f"Added health data with ID {data_id} for user {user_id}")
                
                # If abnormal, trigger an alert entry
                if is_abnormal:
                    message = f"Abnormal health readings detected"
                    details = {k: v for k, v in data.items() if v is not None}
                    
                    self.add_alert(
                        user_id=user_id,
                        message=message,
                        severity="warning",
                        source_agent="health_monitoring_agent",
                        additional_data=json.dumps(details)
                    )
                    
                    logger.warning(f"Abnormal health readings for user {user_id}: {details}")
                
            return True
                
        except ValueError as e:
            logger.error(f"Validation error storing health data for user {user_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error storing health data for user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error storing health data for user {user_id}: {e}")
            raise

    def store_safety_data(self, user_id: int, data: dict) -> bool:
        """
        Store safety data for a user in the database.
        
        Args:
            user_id (int): ID of the user
            data (dict): Safety data to store
                Keys can include: location, movement_type, fall_detected, activity_level,
                time_inactive, risk_level, and any additional_data as JSON
                
        Returns:
            bool: True if data was stored successfully, False otherwise
            
        Raises:
            ValueError: If validation fails
            sqlite3.Error: If database operation fails
        """
        try:
            # Validate inputs
            if not isinstance(user_id, int) or user_id <= 0:
                raise ValueError(f"Invalid user_id: {user_id}")
                
            # Check if user exists
            if not self.get_user(user_id):
                logger.error(f"User with ID {user_id} not found")
                return False
            
            # Extract safety metrics
            location = data.get('location')
            movement_type = data.get('movement_type')
            fall_detected = bool(data.get('fall_detected', False))
            activity_level = data.get('activity_level')
            time_inactive = data.get('time_inactive')
            risk_level = data.get('risk_level')
            
            # Additional data as JSON
            additional_data = data.get('additional_data', {})
            additional_data_json = json.dumps(additional_data) if additional_data else None
            
            # Insert the safety data
            with self.transaction():
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO safety_data (
                    user_id, location, movement_type, fall_detected, 
                    activity_level, time_inactive, risk_level, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, location, movement_type, fall_detected,
                    activity_level, time_inactive, risk_level, additional_data_json
                ))
                
                data_id = cursor.lastrowid
                logger.info(f"Added safety data with ID {data_id} for user {user_id}")
                
                # If fall detected or high risk level, trigger an alert
                if fall_detected or (risk_level and risk_level.lower() == 'high'):
                    severity = "critical" if fall_detected else "warning"
                    message = "Fall detected" if fall_detected else f"High safety risk detected: {risk_level}"
                    details = {k: v for k, v in data.items() if v is not None}
                    
                    self.add_alert(
                        user_id=user_id,
                        message=message,
                        severity=severity,
                        source_agent="safety_monitoring_agent",
                        additional_data=json.dumps(details)
                    )
                    
                    logger.warning(f"Safety alert for user {user_id}: {message} - {details}")
                
            return True
                
        except ValueError as e:
            logger.error(f"Validation error storing safety data for user {user_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error storing safety data for user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error storing safety data for user {user_id}: {e}")
            raise

    def add_alert(self, user_id: int, message: str, severity: str, source_agent: str, 
                    additional_data: str = None) -> int:
        """
        Add an alert to the database.
        
        Args:
            user_id (int): ID of the user the alert is for
            message (str): Alert message text
            severity (str): Alert severity (info, warning, critical)
            source_agent (str): Name of the agent that generated the alert
            additional_data (str, optional): JSON string with additional alert data
            
        Returns:
            int: ID of the new alert
        
        Raises:
            ValueError: If validation fails
            sqlite3.Error: If database operation fails
        """
        try:
            # Validate inputs
            if not isinstance(user_id, int) or user_id <= 0:
                raise ValueError(f"Invalid user_id: {user_id}")
                
            if not message:
                raise ValueError("Alert message cannot be empty")
                
            # Validate severity
            valid_severities = ["info", "warning", "critical"]
            if severity not in valid_severities:
                logger.warning(f"Invalid severity: {severity}. Using 'info' instead.")
                severity = "info"
                
            # Insert the alert
            with self.transaction():
                cursor = self.conn.cursor()
                cursor.execute('''
                INSERT INTO alerts (
                    user_id, message, severity, source_agent, 
                    created_at, additional_data
                ) VALUES (?, ?, ?, ?, datetime('now'), ?)
                ''', (
                    user_id, message, severity, source_agent, additional_data
                ))
                
                alert_id = cursor.lastrowid
                logger.info(f"Added alert with ID {alert_id} for user {user_id}: {message} ({severity})")
                
            return alert_id
                
        except ValueError as e:
            logger.error(f"Validation error adding alert for user {user_id}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error adding alert for user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding alert for user {user_id}: {e}")
            raise