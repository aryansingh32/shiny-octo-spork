import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pyttsx3
import sqlite3
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the correct .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend.env')
load_dotenv(env_path)

# Create directories for models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

class DailyReminderAgent:
    def __init__(self, database):
        self.database = database
        self.acknowledgment_model = None
        self.priority_model = None
        self.optimal_time_model = None
        self.feature_columns = [
            'Type_Appointment', 'Type_Exercise', 'Type_Hydration', 
            'Type_Medication', 'Hour', 'Day_of_Week', 'Is_Weekend', 
            'Priority_Score'
        ]
        
        # Load pre-trained models if they exist
        self._load_models()
        
        # Ensure sample data exists
        if database:
            self._ensure_sample_data_exists()
        
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            # Define model file paths
            acknowledgment_model_path = 'models/acknowledgment_model.pkl'
            priority_model_path = 'models/priority_model.pkl'
            optimal_time_model_path = 'models/optimal_time_model.pkl'
            
            # Check if acknowledgment model exists and load it
            if os.path.exists(acknowledgment_model_path):
                self.acknowledgment_model = joblib.load(acknowledgment_model_path)
                print(f"Loaded Acknowledgment model from {acknowledgment_model_path}")
            else:
                print(f"Acknowledgment model not found at {acknowledgment_model_path}, will train when data is provided")
                
            # Check if priority model exists and load it
            if os.path.exists(priority_model_path):
                self.priority_model = joblib.load(priority_model_path)
                print(f"Loaded Priority model from {priority_model_path}")
            else:
                print(f"Priority model not found at {priority_model_path}, will train when data is provided")
                
            # Check if optimal time model exists and load it
            if os.path.exists(optimal_time_model_path):
                self.optimal_time_model = joblib.load(optimal_time_model_path)
                print(f"Loaded Optimal Time model from {optimal_time_model_path}")
            else:
                print(f"Optimal Time model not found at {optimal_time_model_path}, will train when data is provided")
                
        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
    
    def get_reminders(self, user_id=None, date=None, status=None):
        """
        Get reminders from the database with proper error handling.
        """
        try:
            if not self.database:
                logger.error("Database not initialized")
                return []
            
            # Ensure sample data exists
            self._ensure_sample_data_exists()
                
            # Build query based on filters
            query = "SELECT r.*, u.name as user_name FROM reminders r JOIN users u ON r.user_id = u.id"
            conditions = []
            params = []
            
            if user_id is not None:
                conditions.append("r.user_id = ?")
                params.append(user_id)
                
            if date is not None:
                conditions.append("DATE(r.scheduled_time) = ?")
                params.append(date)
                
            if status is not None:
                conditions.append("r.status = ?")
                params.append(status)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY r.scheduled_time DESC"
                
            # Execute query with error handling
            cursor = self.database.cursor()
            cursor.execute(query, params)
            reminders = cursor.fetchall()
            
            # Convert to list of dictionaries
            reminder_list = []
            for reminder in reminders:
                reminder_dict = dict(reminder)
                if 'scheduled_time' in reminder_dict:
                    if isinstance(reminder_dict['scheduled_time'], datetime):
                        reminder_dict['scheduled_time'] = reminder_dict['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S')
                reminder_list.append(reminder_dict)
                
            logger.info(f"Retrieved {len(reminder_list)} reminders")
            return reminder_list
            
        except Exception as e:
            logger.error(f"Database error while retrieving reminders: {str(e)}")
            return []

    def load_data(self, ml_data_path, cleaned_data_path=None):
        """Load and prepare the datasets for training and validation"""
        # Load the ML-ready dataset
        ml_data = pd.read_csv(ml_data_path, sep=",")
        
        # Define feature columns
        self.feature_columns = [
            'Type_Appointment', 'Type_Exercise', 'Type_Hydration', 
            'Type_Medication', 'Hour', 'Day_of_Week', 'Is_Weekend', 
            'Priority_Score'
        ]
        
        # Convert boolean columns to numeric if they're not
        for col in ['Type_Appointment', 'Type_Exercise', 'Type_Hydration', 'Type_Medication', 'Is_Weekend']:
            if ml_data[col].dtype == bool or ml_data[col].dtype == object:
                ml_data[col] = ml_data[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
        
        print(f"Loaded ML data with {len(ml_data)} records")
        
        # Load the cleaned dataset for additional validation if provided
        cleaned_data = None
        if cleaned_data_path:
            cleaned_data = pd.read_csv(cleaned_data_path)
            print(f"Loaded cleaned data with {len(cleaned_data)} records")
            
            # Convert boolean columns to numeric if needed
            for col in ['Type_Appointment', 'Type_Exercise', 'Type_Hydration', 'Type_Medication', 'Is_Weekend']:
                if col in cleaned_data.columns and (cleaned_data[col].dtype == bool or cleaned_data[col].dtype == object):
                    cleaned_data[col] = cleaned_data[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
            
        return ml_data, cleaned_data
    
    def train_acknowledgment_model(self, data):
        """Train a model to predict if a reminder will be acknowledged"""
        print("\n--- Training Reminder Acknowledgment Model ---")
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data['Acknowledged']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest classifier
        self.acknowledgment_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.acknowledgment_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.acknowledgment_model.predict(X_test)
        print("\nAcknowledgment Model Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Reminder Acknowledgment Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/acknowledgment_confusion_matrix.png')
        
        # Save the model
        joblib.dump(self.acknowledgment_model, 'models/acknowledgment_model.pkl')
        print("Reminder acknowledgment model trained and saved!")
        
        return y_pred
    
    def train_priority_model(self, data):
        """Train a model to predict optimal priority for reminders"""
        print("\n--- Training Reminder Priority Model ---")
        
        # Only proceed if we have Adherence or a similar metric in the data
        if 'Adherence' not in data.columns and 'Priority_Score' not in data.columns:
            print("Warning: No target column for priority prediction found. Skipping priority model.")
            return None
        
        # Use Adherence if available, otherwise use Priority_Score
        target_col = 'Adherence' if 'Adherence' in data.columns else 'Priority_Score'
        
        # Skip rows with missing target values
        valid_data = data.dropna(subset=[target_col])
        if len(valid_data) < len(data):
            print(f"Warning: Dropped {len(data) - len(valid_data)} rows with missing {target_col} values.")
        
        # Prepare features and target
        X = valid_data[self.feature_columns]
        y = valid_data[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest regressor or classifier based on target type
        from sklearn.ensemble import RandomForestRegressor
        if y.dtype == 'object' or len(np.unique(y)) < 10:  # Categorical or small number of unique values
            self.priority_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Numerical
            self.priority_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.priority_model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(self.priority_model, 'models/priority_model.pkl')
        print("Priority prediction model trained and saved!")
        
        return self.priority_model
    
    def train_optimal_time_model(self, data):
        """Train a model to predict optimal time for sending reminders"""
        print("\n--- Training Optimal Reminder Time Model ---")
        
        # Focus on acknowledged reminders for better time prediction
        acknowledged_data = data[data['Acknowledged'] == 1]
        if len(acknowledged_data) < 100:  # Check if we have enough samples
            print("Warning: Not enough acknowledged reminders for optimal time modeling.")
            print(f"Using all data instead ({len(data)} records).")
            acknowledged_data = data
        else:
            print(f"Using {len(acknowledged_data)} acknowledged reminders for optimal time modeling.")
        
        # Prepare features and target (Hour is our target)
        features = [col for col in self.feature_columns if col != 'Hour']
        X = acknowledged_data[features]
        y = acknowledged_data['Hour']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest classifier for hour prediction
        self.optimal_time_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        self.optimal_time_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.optimal_time_model.predict(X_test)
        print("\nOptimal Time Model Accuracy:")
        print(f"Accuracy: {np.mean(y_pred == y_test):.2f}")
        
        # Save the model
        joblib.dump(self.optimal_time_model, 'models/optimal_time_model.pkl')
        print("Optimal reminder time model trained and saved!")
        
        return self.optimal_time_model
    
    def plot_feature_importance(self, model, title):
        """Plot feature importance for a given model"""
        if model is None:
            print(f"Cannot plot feature importance for {title} - model is None")
            return
            
        importance = model.feature_importances_
        
        # Get feature names based on the model
        if title == "Optimal Time Model":
            feature_names = [col for col in self.feature_columns if col != 'Hour']
        else:
            feature_names = self.feature_columns
            
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.title(f'Feature Importance: {title}')
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{title.lower().replace(" ", "_")}.png')
    
    def analyze_reminder_patterns(self, data):
        """Analyze reminder patterns and adherence"""
        print("\n--- Reminder Pattern Analysis ---")
        
        # Create a copy to avoid modifying the original
        analysis_data = data.copy()
        
        # Analyze acknowledgment by reminder type
        plt.figure(figsize=(10, 6))
        reminder_types = ['Type_Appointment', 'Type_Exercise', 'Type_Hydration', 'Type_Medication']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(reminder_types):
            if col in analysis_data.columns:
                sns.countplot(x=col, hue='Acknowledged', data=analysis_data, ax=axes[i])
                axes[i].set_title(f'Acknowledgment by {col}')
                axes[i].set_ylabel('Count')
                axes[i].set_xlabel(col)
                
        plt.tight_layout()
        plt.savefig('results/acknowledgment_by_type.png')
        
        # Analyze acknowledgment by time of day
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Hour', hue='Acknowledged', data=analysis_data)
        plt.title('Reminder Acknowledgment by Hour')
        plt.tight_layout()
        plt.savefig('results/acknowledgment_by_hour.png')
        
        # Analyze acknowledgment by day of week
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Day_of_Week', hue='Acknowledged', data=analysis_data)
        plt.title('Reminder Acknowledgment by Day of Week')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.tight_layout()
        plt.savefig('results/acknowledgment_by_day.png')
        
        # Save analysis results
        analysis_data.to_csv('results/reminder_analysis.csv', index=False)
        
        return analysis_data
    
    def predict_acknowledgment(self, reminder_data):
        """
        Predict the likelihood a reminder will be acknowledged
        
        Args:
            reminder_data: Reminder data to analyze
            
        Returns:
            List of probabilities for acknowledgment
        """
        if not isinstance(reminder_data, pd.DataFrame):
            # Convert to DataFrame if it's a dictionary or list of dictionaries
            if isinstance(reminder_data, dict):
                reminder_data = pd.DataFrame([reminder_data])
            elif isinstance(reminder_data, list) and all(isinstance(item, dict) for item in reminder_data):
                reminder_data = pd.DataFrame(reminder_data)
            else:
                logger.error("Unsupported data format for acknowledgment prediction")
                return []
                
        # Process reminder data to extract features
        features = self._extract_reminder_features(reminder_data)
        
        # Use acknowledgment model if available
        if self.acknowledgment_model is not None:
            # Make sure we have all required features
            missing_cols = [col for col in self.feature_columns if col not in features.columns]
            for col in missing_cols:
                features[col] = 0  # Default value
                
            X = features[self.feature_columns]
            
            # Get raw probabilities for acknowledgment
            if hasattr(self.acknowledgment_model, 'predict_proba'):
                probabilities = self.acknowledgment_model.predict_proba(X)[:, 1]  # Probability of class 1
            else:
                # If model doesn't support probabilities, use binary prediction
                predictions = self.acknowledgment_model.predict(X)
                probabilities = predictions.astype(float)
                
            return probabilities.tolist()
        else:
            # Fallback to rule-based approach
            acknowledgment_probs = []
            
            # Rule 1: Time of day affects acknowledgment
            if 'Hour' in features.columns:
                # Higher acknowledgment in morning and evening
                hour_effect = np.zeros(len(features))
                
                # Morning (7-10 AM): High acknowledgment
                morning_mask = (features['Hour'] >= 7) & (features['Hour'] <= 10)
                hour_effect[morning_mask] = 0.8
                
                # Mid-day (11-4 PM): Moderate acknowledgment
                midday_mask = (features['Hour'] >= 11) & (features['Hour'] <= 16) 
                hour_effect[midday_mask] = 0.6
                
                # Evening (5-9 PM): High acknowledgment
                evening_mask = (features['Hour'] >= 17) & (features['Hour'] <= 21)
                hour_effect[evening_mask] = 0.75
                
                # Night (10 PM-6 AM): Low acknowledgment
                night_mask = (features['Hour'] >= 22) | (features['Hour'] <= 6)
                hour_effect[night_mask] = 0.3
                
                acknowledgment_probs.append(hour_effect)
            
            # Rule 2: Reminder type affects acknowledgment
            if 'reminder_type' in reminder_data.columns:
                type_effect = np.zeros(len(features))
                
                # Medication: Higher acknowledgment
                med_mask = reminder_data['reminder_type'] == 'medication'
                type_effect[med_mask] = 0.8
                
                # Appointment: High acknowledgment
                appt_mask = reminder_data['reminder_type'] == 'appointment'
                type_effect[appt_mask] = 0.75
                
                # Activity: Moderate acknowledgment
                activity_mask = reminder_data['reminder_type'] == 'activity'
                type_effect[activity_mask] = 0.6
                
                # Other: Lower acknowledgment
                other_mask = ~(med_mask | appt_mask | activity_mask)
                type_effect[other_mask] = 0.5
                
                acknowledgment_probs.append(type_effect)
            
            # Rule 3: Priority affects acknowledgment
            if 'priority' in reminder_data.columns:
                priority_effect = np.zeros(len(features))
                
                # High priority: High acknowledgment
                high_mask = reminder_data['priority'] == 'high'
                priority_effect[high_mask] = 0.85
                
                # Medium priority: Moderate acknowledgment
                medium_mask = reminder_data['priority'] == 'medium'
                priority_effect[medium_mask] = 0.7
                
                # Low priority: Lower acknowledgment
                low_mask = reminder_data['priority'] == 'low'
                priority_effect[low_mask] = 0.5
                
                acknowledgment_probs.append(priority_effect)
                
            # Combine all factors, or use default if none available
            if acknowledgment_probs:
                combined_probs = np.mean(acknowledgment_probs, axis=0)
                return combined_probs.tolist()
            else:
                # Default acknowledgment probability (medium likelihood)
                return [0.65] * len(reminder_data)
                
    def predict_priority(self, reminder_data):
        """
        Predict optimal priority for reminders
        
        Args:
            reminder_data: Reminder data to analyze
            
        Returns:
            List of priority classes ("low", "medium", "high")
        """
        if not isinstance(reminder_data, pd.DataFrame):
            # Convert to DataFrame if necessary
            if isinstance(reminder_data, dict):
                reminder_data = pd.DataFrame([reminder_data])
            elif isinstance(reminder_data, list) and all(isinstance(item, dict) for item in reminder_data):
                reminder_data = pd.DataFrame(reminder_data)
            else:
                logger.error("Unsupported data format for priority prediction")
                return []
                
        # If priority is already in the data, just return it
        if 'priority' in reminder_data.columns:
            return reminder_data['priority'].tolist()
                
        # Process reminder data to extract features
        features = self._extract_reminder_features(reminder_data)
        
        # Use priority model if available
        if self.priority_model is not None:
            # Make sure we have all required features
            missing_cols = [col for col in self.feature_columns if col not in features.columns]
            for col in missing_cols:
                features[col] = 0  # Default value
                
            X = features[self.feature_columns]
            
            # Predict priority
            priority_values = self.priority_model.predict(X)
            
            # Convert numeric priority to labels
            priority_labels = []
            for val in priority_values:
                if isinstance(val, (int, float)):
                    if val <= 1:
                        priority_labels.append("low")
                    elif val <= 2:
                        priority_labels.append("medium")
                    else:
                        priority_labels.append("high")
                else:
                    priority_labels.append(val)
                    
            return priority_labels
        else:
            # Fallback to rule-based approach
            priority_values = []
            
            # Rule 1: Reminder type affects priority
            if 'reminder_type' in reminder_data.columns:
                # Set priority based on type
                for r_type in reminder_data['reminder_type']:
                    if r_type == 'medication':
                        priority_values.append("high")
                    elif r_type == 'appointment':
                        priority_values.append("medium")
                    elif r_type == 'activity':
                        priority_values.append("medium")
                    else:
                        priority_values.append("low")
                        
                return priority_values
            else:
                # Default to medium priority if no type information
                return ["medium"] * len(reminder_data)
                
    def predict_optimal_time(self, reminder_data):
        """
        Predict optimal timing for reminders
        
        Args:
            reminder_data: Reminder data to analyze
            
        Returns:
            List of optimal hour values (0-23)
        """
        if not isinstance(reminder_data, pd.DataFrame):
            # Convert to DataFrame if necessary
            if isinstance(reminder_data, dict):
                reminder_data = pd.DataFrame([reminder_data])
            elif isinstance(reminder_data, list) and all(isinstance(item, dict) for item in reminder_data):
                reminder_data = pd.DataFrame(reminder_data)
            else:
                logger.error("Unsupported data format for optimal time prediction")
                return []
                
        # Process reminder data to extract features
        features = self._extract_reminder_features(reminder_data)
        
        # Use optimal time model if available
        if self.optimal_time_model is not None:
            # Make sure we have all required features
            missing_cols = [col for col in self.feature_columns if col not in features.columns]
            for col in missing_cols:
                features[col] = 0  # Default value
                
            X = features[self.feature_columns]
            
            # Predict optimal hour
            optimal_hours = self.optimal_time_model.predict(X)
            
            # Ensure hours are valid (0-23)
            optimal_hours = np.clip(optimal_hours, 0, 23).astype(int)
                    
            return optimal_hours.tolist()
        else:
            # Fallback to rule-based approach based on reminder type
            optimal_hours = []
            
            if 'reminder_type' in reminder_data.columns:
                for r_type in reminder_data['reminder_type']:
                    if r_type == 'medication':
                        # Medications often taken with meals
                        if np.random.random() < 0.4:
                            optimal_hours.append(8)  # Breakfast
                        elif np.random.random() < 0.7:
                            optimal_hours.append(12)  # Lunch
                        else:
                            optimal_hours.append(18)  # Dinner
                    elif r_type == 'appointment':
                        # Appointments typically mid-day
                        optimal_hours.append(np.random.choice([10, 11, 14, 15]))
                    elif r_type == 'activity':
                        # Activities often in morning or evening
                        if np.random.random() < 0.6:
                            optimal_hours.append(9)  # Morning
                        else:
                            optimal_hours.append(17)  # Evening
                    else:
                        # Default to mid-day for other types
                        optimal_hours.append(14)
            else:
                # Default times if no type information
                for _ in range(len(reminder_data)):
                    optimal_hours.append(np.random.choice([9, 12, 16]))
                    
            return optimal_hours
            
    def _extract_reminder_features(self, reminder_data):
        """
        Extract relevant features for reminder predictions
        
        Args:
            reminder_data: Reminder data
            
        Returns:
            DataFrame with extracted features
        """
        # Create empty DataFrame to hold features
        features = pd.DataFrame(index=reminder_data.index)
        
        # Extract hour from scheduled_time if available
        if 'scheduled_time' in reminder_data.columns:
            try:
                # Try to convert to datetime
                scheduled_times = pd.to_datetime(reminder_data['scheduled_time'])
                features['Hour'] = scheduled_times.dt.hour
                features['Day_of_Week'] = scheduled_times.dt.dayofweek
                features['Is_Weekend'] = (scheduled_times.dt.dayofweek >= 5).astype(int)
            except:
                # If conversion fails, set defaults
                features['Hour'] = 12  # Default to noon
                features['Day_of_Week'] = 0  # Default to Monday
                features['Is_Weekend'] = 0  # Default to weekday
        else:
            # Default values if scheduled_time not available
            features['Hour'] = 12
            features['Day_of_Week'] = 0
            features['Is_Weekend'] = 0
            
        # One-hot encode reminder types
        if 'reminder_type' in reminder_data.columns:
            # Create indicator variables for common types
            features['Type_Medication'] = (reminder_data['reminder_type'] == 'medication').astype(int)
            features['Type_Appointment'] = (reminder_data['reminder_type'] == 'appointment').astype(int)
            features['Type_Exercise'] = (reminder_data['reminder_type'] == 'activity').astype(int)
            features['Type_Hydration'] = (reminder_data['reminder_type'] == 'hydration').astype(int)
        else:
            # Default values if type not available
            features['Type_Medication'] = 0
            features['Type_Appointment'] = 0
            features['Type_Exercise'] = 0
            features['Type_Hydration'] = 0
            
        # Convert priority to numeric score
        if 'priority' in reminder_data.columns:
            # Map string priorities to numeric scores
            priority_map = {'low': 1, 'medium': 2, 'high': 3}
            features['Priority_Score'] = reminder_data['priority'].map(lambda p: priority_map.get(p, 2))
        else:
            # Default to medium priority
            features['Priority_Score'] = 2
            
        return features
    
    def suggest_optimal_schedule(self, upcoming_reminders):
        """
        Suggest optimal schedule for a batch of upcoming reminders and send voice reminders.
        """
        if not isinstance(upcoming_reminders, pd.DataFrame):
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(upcoming_reminders, list) and all(isinstance(item, dict) for item in upcoming_reminders):
                upcoming_reminders = pd.DataFrame(upcoming_reminders)

        results = []
        for i, reminder in upcoming_reminders.iterrows():
            # Extract basic reminder info
            reminder_data = {
                'original_info': {
                    'reminder_type': self._get_reminder_type(reminder),
                    'scheduled_time': f"{int(reminder['Hour']):02d}:00",
                    'day_of_week': int(reminder['Day_of_Week'])
                }
            }

            # Predict acknowledgment with current schedule
            ack_pred = self.predict_acknowledgment(pd.DataFrame([reminder]))
            reminder_data['current_prediction'] = ack_pred

            # Deliver acknowledgment prediction via voice
            acknowledgment_text = (
                f"The reminder for {reminder_data['original_info']['reminder_type']} scheduled at "
                f"{reminder_data['original_info']['scheduled_time']} has an acknowledgment probability of "
                f"{ack_pred['acknowledgment_probability'] * 100:.1f}%."
            )
            self.send_voice_reminder(acknowledgment_text)

            # If unlikely to be acknowledged, suggest optimizations
            if not ack_pred['will_acknowledge']:
                # Copy reminder for modification
                modified = reminder.copy()

                # Find optimal hour if model is available
                if self.optimal_time_model is not None:
                    features = [col for col in self.feature_columns if col != 'Hour']
                    X_pred = pd.DataFrame([reminder[features]])
                    optimal_hour = self.optimal_time_model.predict(X_pred)[0]
                    modified['Hour'] = optimal_hour

                    # Check if new time improves acknowledgment
                    new_pred = self.predict_acknowledgment(pd.DataFrame([modified]))
                    if new_pred['acknowledgment_probability'] > ack_pred['acknowledgment_probability']:
                        reminder_data['optimized'] = {
                            'suggested_time': f"{int(optimal_hour):02d}:00",
                            'new_probability': new_pred['acknowledgment_probability']
                        }
                        optimization_text = (
                            f"Consider rescheduling the reminder for {reminder_data['original_info']['reminder_type']} to "
                            f"{int(optimal_hour):02d}:00 to improve acknowledgment probability to "
                            f"{new_pred['acknowledgment_probability'] * 100:.1f}%."
                        )
                        self.send_voice_reminder(optimization_text)

            # Send voice reminder
            reminder_text = f"Reminder: {reminder_data['original_info']['reminder_type']} scheduled at {reminder_data['original_info']['scheduled_time']}."
            self.send_voice_reminder(reminder_text)

            results.append(reminder_data)

        return results
    
    def _get_reminder_type(self, reminder):
        """Helper to determine reminder type from boolean columns"""
        type_cols = {
            'Type_Appointment': 'Appointment',
            'Type_Exercise': 'Exercise',
            'Type_Hydration': 'Hydration',
            'Type_Medication': 'Medication'
        }
        
        for col, label in type_cols.items():
            if col in reminder and reminder[col] == 1:
                return label
        
        return "Other"
    
    def send_voice_reminder(self, reminder_text, user_id=None, reminder_id=None, priority=None):
        """
        Generate and play a voice reminder using the cross-platform voice reminder service.
        
        Args:
            reminder_text (str): The text of the reminder to be spoken.
            user_id (int, optional): The user ID associated with this reminder.
            reminder_id (int, optional): The reminder ID in the database.
            priority (str, optional): Priority level of the reminder (high, medium, low).
        """
        try:
            # Import the voice reminder service
            from voice_reminder_service import get_voice_reminder_service
            
            # Settings based on priority
            settings = {}
            if priority:
                if priority.lower() == 'high':
                    settings = {
                        'rate': 140,  # Slightly slower for emphasis
                        'volume': 1.0,  # Full volume
                        'play_audio': True,
                        'save_audio': True
                    }
                elif priority.lower() == 'medium':
                    settings = {
                        'rate': 150,  # Normal rate
                        'volume': 0.9,  # Slightly lower volume
                        'play_audio': True,
                        'save_audio': True
                    }
                elif priority.lower() == 'low':
                    settings = {
                        'rate': 160,  # Slightly faster
                        'volume': 0.8,  # Lower volume
                        'play_audio': True,
                        'save_audio': True
                    }
            
            # Get the voice service
            voice_service = get_voice_reminder_service()
            
            # Queue the reminder
            success = voice_service.queue_reminder(
                text=reminder_text,
                user_id=user_id,
                reminder_id=reminder_id,
                settings=settings,
                callback=self._on_voice_reminder_complete
            )
            
            if success:
                logger.info(f"Voice reminder queued: {reminder_text}")
            else:
                logger.warning(f"Failed to queue voice reminder: {reminder_text}")
                
            return success
        except ImportError:
            # Fall back to pyttsx3 if voice service not available
            try:
                # Initialize the text-to-speech engine
                tts_engine = pyttsx3.init()
                
                # Set properties (optional)
                tts_engine.setProperty('rate', 150)  # Speed of speech
                tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
                
                # Speak the reminder text
                tts_engine.say(reminder_text)
                tts_engine.runAndWait()
                
                logger.info(f"Voice reminder sent using fallback method: {reminder_text}")
                return True
            except Exception as e:
                logger.error(f"Error sending voice reminder using fallback method: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error sending voice reminder: {str(e)}")
            return False
            
    def _on_voice_reminder_complete(self, success, audio_path=None, error=None):
        """
        Callback function called when a voice reminder has been processed.
        
        Args:
            success (bool): Whether the reminder was successfully spoken/saved
            audio_path (str, optional): Path to the saved audio file
            error (str, optional): Error message if unsuccessful
        """
        if success:
            logger.info(f"Voice reminder completed successfully. Audio saved to: {audio_path}")
            
            # If we have a database and audio path, we could save the audio path
            # to the database for reference in the UI
            if self.database and audio_path:
                try:
                    # This would need a schema that supports audio_path
                    pass
                except Exception as e:
                    logger.error(f"Error storing voice reminder audio path: {str(e)}")
        else:
            logger.error(f"Voice reminder failed: {error}")
    
    def speak_reminder_by_id(self, reminder_id):
        """
        Speak a specific reminder by its ID.
        
        Args:
            reminder_id (int): The ID of the reminder to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.database:
                logger.error("Database not initialized")
                return False
                
            # Get the reminder from the database
            cursor = self.database.cursor()
            cursor.execute("""
                SELECT r.*, u.name as user_name 
                FROM reminders r
                JOIN users u ON r.user_id = u.id
                WHERE r.id = ?
            """, (reminder_id,))
            reminder = cursor.fetchone()
            
            if not reminder:
                logger.error(f"Reminder with ID {reminder_id} not found")
                return False
                
            # Convert to dictionary
            reminder_dict = dict(reminder)
            
            # Format reminder text
            user_name = reminder_dict.get('user_name', 'User')
            title = reminder_dict.get('title', 'Reminder')
            description = reminder_dict.get('description', '')
            priority = reminder_dict.get('priority', 'medium')
            
            # Create human-friendly reminder text
            reminder_text = f"Hello {user_name}, this is a {priority} priority reminder: {title}."
            if description:
                reminder_text += f" {description}"
                
            # Send the voice reminder
            return self.send_voice_reminder(
                reminder_text=reminder_text,
                user_id=reminder_dict.get('user_id'),
                reminder_id=reminder_id,
                priority=priority
            )
                
        except Exception as e:
            logger.error(f"Error speaking reminder by ID: {str(e)}")
            return False

    def get_upcoming_reminders(self, days=7):
        """
        Get upcoming reminders for the next N days.
        
        Args:
            days (int): Number of days to look ahead
            
        Returns:
            list: List of upcoming reminders
        """
        try:
            if not self.database:
                logger.error("Database not initialized")
                return []
                
            # Ensure sample data exists
            self._ensure_sample_data_exists()
                
            # Calculate the date range
            now = datetime.now()
            end_date = (now + timedelta(days=days)).strftime('%Y-%m-%d')
            today = now.strftime('%Y-%m-%d')
            
            # Build query for upcoming reminders
            query = """
                SELECT r.*, u.name as user_name 
                FROM reminders r
                JOIN users u ON r.user_id = u.id
                WHERE DATE(r.scheduled_time) BETWEEN ? AND ?
                AND (r.status = 'pending' OR r.status = 'scheduled')
                ORDER BY r.scheduled_time ASC
            """
            
            # Execute query with error handling
            cursor = self.database.cursor()
            cursor.execute(query, (today, end_date))
            reminders = cursor.fetchall()
            
            # Convert to list of dictionaries
            reminder_list = []
            for reminder in reminders:
                reminder_dict = dict(reminder)
                if 'scheduled_time' in reminder_dict:
                    if isinstance(reminder_dict['scheduled_time'], datetime):
                        reminder_dict['scheduled_time'] = reminder_dict['scheduled_time'].strftime('%Y-%m-%d %H:%M:%S')
                reminder_list.append(reminder_dict)
                
            logger.info(f"Retrieved {len(reminder_list)} upcoming reminders")
            return reminder_list
            
        except Exception as e:
            logger.error(f"Error retrieving upcoming reminders: {str(e)}")
            return []

    def _ensure_sample_data_exists(self):
        """
        Ensure sample reminder data exists in the database.
        If no reminders exist, create some sample data.
        """
        try:
            cursor = self.database.cursor()
            
            # Check if reminders table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reminders'")
            if not cursor.fetchone():
                # Create reminders table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        reminder_type TEXT,
                        scheduled_time TEXT NOT NULL,
                        priority TEXT,
                        recurrence TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
                self.database.commit()
            
            # Check if users table exists and has at least one user
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count == 0:
                # Add a sample user if none exist
                cursor.execute("""
                    INSERT INTO users (name, age, preferences)
                    VALUES (?, ?, ?)
                """, ("John Doe", 65, '{"theme":"light"}'))
                self.database.commit()
            
            # Get existing user IDs
            cursor.execute("SELECT id FROM users LIMIT 3")
            user_ids = [row[0] for row in cursor.fetchall()]
            
            # Check if any reminders exist
            cursor.execute("SELECT COUNT(*) FROM reminders")
            reminder_count = cursor.fetchone()[0]
            
            if reminder_count == 0 and user_ids:
                # Current time
                now = datetime.now()
                
                # Sample reminders data
                sample_reminders = [
                    # Today's reminders
                    {
                        "user_id": user_ids[0],
                        "title": "Take Medication",
                        "description": "Remember to take your heart medication with food",
                        "reminder_type": "medication",
                        "scheduled_time": (now + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                        "priority": "high",
                        "recurrence": "daily",
                        "status": "pending"
                    },
                    {
                        "user_id": user_ids[0],
                        "title": "Doctor Appointment",
                        "description": "Checkup with Dr. Smith",
                        "reminder_type": "appointment",
                        "scheduled_time": (now + timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),
                        "priority": "medium",
                        "recurrence": "once",
                        "status": "pending"
                    },
                    
                    # Tomorrow's reminders
                    {
                        "user_id": user_ids[0],
                        "title": "Exercise Time",
                        "description": "Light walking exercise for 20 minutes",
                        "reminder_type": "activity",
                        "scheduled_time": (now + timedelta(days=1, hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                        "priority": "medium",
                        "recurrence": "daily",
                        "status": "pending"
                    },
                    
                    # Next week reminders
                    {
                        "user_id": user_ids[0],
                        "title": "Family Visit",
                        "description": "Your daughter will visit",
                        "reminder_type": "social",
                        "scheduled_time": (now + timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S'),
                        "priority": "low",
                        "recurrence": "once",
                        "status": "pending"
                    }
                ]
                
                # Insert sample reminders
                for reminder in sample_reminders:
                    cursor.execute("""
                        INSERT INTO reminders 
                        (user_id, title, description, reminder_type, scheduled_time, priority, recurrence, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        reminder["user_id"], 
                        reminder["title"], 
                        reminder["description"], 
                        reminder["reminder_type"], 
                        reminder["scheduled_time"], 
                        reminder["priority"], 
                        reminder["recurrence"], 
                        reminder["status"]
                    ))
                
                self.database.commit()
                logger.info("Created sample reminder data")
                
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")


def main():
    # Initialize the agent with database
    db_path = os.getenv("DATABASE_PATH", "elderly_care.db")
    database = sqlite3.connect(db_path, check_same_thread=False)
    database.row_factory = sqlite3.Row
    
    agent = DailyReminderAgent(database)
    
    try:
        # Load and prepare data
        ml_data_path = "daily_reminder_ml_ready.csv"
        cleaned_data_path = "daily_reminder_cleaned.csv"
        
        ml_data, cleaned_data = agent.load_data(ml_data_path, cleaned_data_path)
        
        # Train models
        agent.train_acknowledgment_model(ml_data)
        agent.train_priority_model(ml_data)
        agent.train_optimal_time_model(ml_data)
        
        # Analyze patterns
        agent.analyze_reminder_patterns(ml_data)
        
        print("Daily Reminder Agent training complete!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        database.close()


if __name__ == "__main__":
    main()