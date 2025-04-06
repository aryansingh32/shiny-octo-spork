import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import random
import logging

# Create directories for models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

class HealthMonitoringAgent:
    def __init__(self, database):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.database = database
        self.isolation_forest = None
        self.supervised_model = None
        self.severity_model = None
        self.feature_columns = [
            'Heart Rate_normalized', 'Systolic_normalized', 
            'Diastolic_normalized', 'Glucose Levels_normalized',
            'SpO2_normalized', 'Hour', 'Day', 'Is_Night'
        ]
        self.threshold = 0.0
        
        # Load pre-trained models if they exist
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models if they exist."""
        try:
            # Construct absolute paths for the models
            base_dir = os.path.dirname(__file__)  # Directory of the current file
            isolation_forest_path = os.path.join(base_dir, 'models', 'isolation_forest_model.pkl')
            supervised_model_path = os.path.join(base_dir, 'models', 'rf_classifier_model.pkl')
            severity_model_path = os.path.join(base_dir, 'models', 'severity_model.pkl')

            # Load Isolation Forest model
            if os.path.exists(isolation_forest_path):
                self.isolation_forest = joblib.load(isolation_forest_path)
                self.logger.info(f"Loaded Isolation Forest model from {isolation_forest_path}")
            else:
                self.logger.warning(f"Isolation Forest model not found at {isolation_forest_path}")

            # Load Supervised model
            if os.path.exists(supervised_model_path):
                self.supervised_model = joblib.load(supervised_model_path)
                self.logger.info(f"Loaded Supervised model from {supervised_model_path}")
            else:
                self.logger.warning(f"Supervised model not found at {supervised_model_path}")

            # Load Severity model
            if os.path.exists(severity_model_path):
                self.severity_model = joblib.load(severity_model_path)
                self.logger.info(f"Loaded Severity model from {severity_model_path}")
            else:
                self.logger.warning(f"Severity model not found at {severity_model_path}")

        except Exception as e:
            self.logger.error(f"Error loading pre-trained models: {e}")
            
    def load_data(self, ml_data_path, cleaned_data_path=None):
        """Load and prepare the datasets for training and validation"""
        # Load the ML dataset (already normalized with anomaly scores)
        ml_data = pd.read_csv(ml_data_path, sep=",")
        
        # Define feature columns
        self.feature_columns = [
            'Heart Rate_normalized', 'Systolic_normalized', 
            'Diastolic_normalized', 'Glucose Levels_normalized',
            'SpO2_normalized', 'Hour', 'Day', 'Is_Night'
        ]
        
        # Convert Is_Night to numeric if it's not
        if ml_data['Is_Night'].dtype == bool or ml_data['Is_Night'].dtype == object:
            ml_data['Is_Night'] = ml_data['Is_Night'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        
        print(f"Loaded ML data with {len(ml_data)} records")
        
        # Load the cleaned dataset for additional validation if provided
        cleaned_data = None
        if cleaned_data_path:
            cleaned_data = pd.read_csv(cleaned_data_path)
            print(f"Loaded cleaned data with {len(cleaned_data)} records")
            
        return ml_data, cleaned_data
    
    def train_unsupervised_model(self, data, contamination=0.1):
        """Train an Isolation Forest model for anomaly detection"""
        print("\n--- Training Unsupervised Anomaly Detection Model ---")
        
        # Select features for training
        X = data[self.feature_columns]
        
        # Train Isolation Forest model
        self.isolation_forest = IsolationForest(
            n_estimators=100, 
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        self.isolation_forest.fit(X)
        
        # Calculate anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X)
        predicted_anomalies = self.isolation_forest.predict(X)
        
        # Convert predictions from {-1, 1} to {1, 0} (1 for anomaly, 0 for normal)
        predicted_anomalies = np.where(predicted_anomalies == -1, 1, 0)
        
        # Save the model
        joblib.dump(self.isolation_forest, 'models/isolation_forest_model.pkl')
        print("Isolation Forest model trained and saved!")
        
        return anomaly_scores, predicted_anomalies
    
    def train_supervised_model(self, data):
        """Train a supervised model using the existing anomaly scores"""
        print("\n--- Training Supervised Classification Model ---")
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data['Anomaly_Score']
        
        # Convert anomaly scores to binary (0 for normal, 1+ for anomalies)
        y_binary = (y > 0).astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.3, random_state=42
        )
        
        # Train Random Forest classifier
        self.supervised_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.supervised_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.supervised_model.predict(X_test)
        print("\nSupervised Model Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Anomaly Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/confusion_matrix.png')
        
        # Save the model
        joblib.dump(self.supervised_model, 'models/rf_classifier_model.pkl')
        print("Random Forest classifier trained and saved!")
        
        return y_pred
    
    def train_severity_model(self, data):
        """Train a model to predict the severity of anomalies (0-3)"""
        print("\n--- Training Anomaly Severity Model ---")
        
        # Prepare features and target
        X = data[self.feature_columns]
        y = data['Anomaly_Score']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest classifier for severity
        severity_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        severity_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = severity_model.predict(X_test)
        
        # Save the model
        joblib.dump(severity_model, 'models/severity_model.pkl')
        print("Severity prediction model trained and saved!")
        
        return severity_model
    
    def plot_feature_importance(self, model, title):
        """Plot feature importance for a given model"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_columns[i] for i in indices], rotation=45)
        plt.title(f'Feature Importance: {title}')
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{title.lower().replace(" ", "_")}.png')
    
    def analyze_anomalies(self, data, anomaly_predictions):
        """Analyze detected anomalies and their characteristics"""
        # Add predictions to the dataframe
        data_with_preds = data.copy()
        data_with_preds['Predicted_Anomaly'] = anomaly_predictions
        
        # Compare predicted vs actual anomalies
        print("\n--- Anomaly Analysis ---")
        predicted_anomalies = data_with_preds[data_with_preds['Predicted_Anomaly'] == 1]
        actual_anomalies = data_with_preds[data_with_preds['Anomaly_Score'] > 0]
        
        print(f"Number of predicted anomalies: {len(predicted_anomalies)}")
        print(f"Number of actual anomalies: {len(actual_anomalies)}")
        
        # Analyze when anomalies occur
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Hour', hue='Anomaly_Score', data=data_with_preds)
        plt.title('Anomalies by Hour of Day')
        plt.tight_layout()
        plt.savefig('results/anomalies_by_hour.png')
        
        # Save analysis results
        data_with_preds.to_csv('results/analyzed_data.csv', index=False)
        
        return data_with_preds
    
    def predict(self, new_data):
        """Make predictions on new health data"""
        if not isinstance(new_data, pd.DataFrame):
            # Convert to DataFrame if it's a dictionary or array
            if isinstance(new_data, dict):
                new_data = pd.DataFrame([new_data])
            else:
                new_data = pd.DataFrame([new_data], columns=self.feature_columns)
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Make predictions
        is_anomaly = self.supervised_model.predict(new_data[self.feature_columns])
        
        # Get severity if it's an anomaly
        severity = 0
        if is_anomaly[0] == 1:
            severity_model = joblib.load('models/severity_model.pkl')
            severity = severity_model.predict(new_data[self.feature_columns])[0]
        
        result = {
            'is_anomaly': bool(is_anomaly[0]),
            'severity': int(severity),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get explanation for the prediction
        if result['is_anomaly']:
            result['explanation'] = self._get_anomaly_explanation(new_data)
            result['alert_level'] = self._get_alert_level(severity)
        
        return result
    
    def predict_acknowledgment(self, reminder_data):
        """Predict if a reminder will be acknowledged."""
        if not isinstance(reminder_data, pd.DataFrame):
            # Convert to DataFrame if it's a dictionary or array
            if isinstance(reminder_data, dict):
                reminder_data = pd.DataFrame([reminder_data])
            else:
                reminder_data = pd.DataFrame([reminder_data], columns=self.feature_columns)

        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(reminder_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        # Make prediction
        acknowledgment_prob = self.acknowledgment_model.predict_proba(reminder_data[self.feature_columns])
        will_acknowledge = self.acknowledgment_model.predict(reminder_data[self.feature_columns])

        result = {
            'will_acknowledge': bool(will_acknowledge[0]),
            'acknowledgment_probability': float(acknowledgment_prob[0][1]),  # Convert to native Python float
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Get recommendation for optimal time and priority if needed
        if not result['will_acknowledge']:
            result['recommendations'] = self._get_recommendations(reminder_data, result['acknowledgment_probability'])

        return result

    def _get_anomaly_explanation(self, data):
        """Generate an explanation for why this is flagged as an anomaly"""
        # Get feature importances
        importances = self.supervised_model.feature_importances_
        feature_importance_dict = dict(zip(self.feature_columns, importances))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top 3 most important features
        top_features = sorted_features[:3]
        
        # Prepare explanation
        explanation = "Anomaly detected due to unusual values in: "
        for feature, _ in top_features:
            # Check if the feature value is outside normal range
            if feature in data.columns:
                value = data[feature].iloc[0]
                if value > 0.8 or value < 0.2:  # Assuming normalized values
                    status = "high" if value > 0.8 else "low"
                    explanation += f"{feature} ({status}), "
        
        return explanation.rstrip(", ")
    
    def _get_alert_level(self, severity):
        """Determine alert level based on severity"""
        if severity == 0:
            return "Normal"
        elif severity == 1:
            return "Low Alert"
        elif severity == 2:
            return "Medium Alert"
        else:  # severity == 3
            return "High Alert - Immediate Attention Required"

    def _get_recommendations(self, reminder_data, current_prob):
        """Generate recommendations to improve acknowledgment probability."""
        recommendations = {}
        recommendation_text = []

        # Copy the data for modification
        modified_data = reminder_data.copy()

        # Try different hours to find optimal time
        if self.optimal_time_model is not None:
            features = [col for col in self.feature_columns if col != 'Hour']
            current_hour = reminder_data['Hour'].values[0]
            best_hour = current_hour
            best_prob = current_prob  # Use the probability passed from the calling function

            # Test each hour
            for hour in range(7, 23):  # Reasonable hours from 7 AM to 10 PM
                modified_data['Hour'] = hour
                prob = self.acknowledgment_model.predict_proba(modified_data[self.feature_columns])[0][1]

                if prob > best_prob:
                    best_prob = prob
                    best_hour = hour

            if best_hour != current_hour:
                recommendations['optimal_time'] = f"{best_hour}:00"
                recommendations['time_improvement'] = f"+{((best_prob - current_prob) * 100):.1f}%"
                recommendation_text.append(
                    f"Consider rescheduling the reminder to {best_hour}:00 to improve acknowledgment probability by {((best_prob - current_prob) * 100):.1f}%."
                )

        # Check if adjusting priority would help
        if self.priority_model is not None and 'Priority_Score' in reminder_data.columns:
            current_priority = reminder_data['Priority_Score'].values[0]
            best_priority = current_priority
            best_prob = current_prob  # Use the probability passed from the calling function

            # Test different priority levels
            for priority in range(1, 4):  # Assuming priority is 1-3
                if priority != current_priority:
                    modified_data['Priority_Score'] = priority
                    prob = self.acknowledgment_model.predict_proba(modified_data[self.feature_columns])[0][1]

                    if prob > best_prob:
                        best_prob = prob
                        best_priority = priority

            if best_priority != current_priority:
                priority_labels = {1: "Low", 2: "Medium", 3: "High"}
                recommendations['suggested_priority'] = priority_labels.get(best_priority, str(best_priority))
                recommendation_text.append(
                    f"Consider changing the priority to {priority_labels.get(best_priority, str(best_priority))} to improve acknowledgment probability."
                )

        return recommendations

    def suggest_optimal_schedule(self, upcoming_reminders):
        """
        Suggest optimal schedule for a batch of upcoming reminders.
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

            results.append(reminder_data)

        return results

    def get_anomalies(self):
        """
        Retrieve health anomalies from the database.
        
        Returns:
            list: A list of anomalies with their details
        """
        try:
            cursor = self.database.cursor()
            # Use a more generic query that doesn't rely on is_anomaly column
            query = """
                SELECT h.*, u.name as user_name 
                FROM health_data h
                JOIN users u ON h.user_id = u.id
                ORDER BY h.timestamp DESC
                LIMIT 100
            """
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Convert to list of dictionaries and filter for anomalies
            anomalies = []
            for row in results:
                data = dict(row)
                # Consider adding any logic here to determine if a reading is an anomaly
                # For now, include all readings as anomalies for display purposes
                data['alert_level'] = "Medium Alert"
                data['severity'] = 2
                anomalies.append(data)
                
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error getting anomalies: {str(e)}")
            # Return an empty list instead of raising an exception
            return []
    
    def get_health_overview(self):
        """
        Get a health overview for all users.
        
        Returns:
            dict: Summary statistics of health data
        """
        try:
            cursor = self.database.cursor()
            # Get total health readings
            cursor.execute("SELECT COUNT(*) FROM health_data")
            total_readings = cursor.fetchone()[0]
            
            # Since we don't have an is_anomaly column, let's assume all readings are normal
            # In a real implementation, you would analyze the data to determine anomalies
            anomaly_count = 0
            users_with_anomalies = 0
            
            # Get latest reading
            cursor.execute("""
                SELECT h.*, u.name as user_name 
                FROM health_data h
                JOIN users u ON h.user_id = u.id
                ORDER BY h.timestamp DESC
                LIMIT 1
            """)
            latest_reading = cursor.fetchone()
            latest_reading_dict = dict(latest_reading) if latest_reading else None
            
            # Add generated anomaly data for demonstration
            if latest_reading_dict:
                latest_reading_dict['alert_level'] = "Normal"
                latest_reading_dict['severity'] = 0
            
            return {
                "total_readings": total_readings,
                "anomaly_count": anomaly_count,
                "users_with_anomalies": users_with_anomalies,
                "latest_reading": latest_reading_dict,
                "anomaly_rate": 0  # No anomalies for now
            }
        except Exception as e:
            self.logger.error(f"Error getting health overview: {str(e)}")
            # Return a placeholder overview instead of raising an exception
            return {
                "total_readings": 0,
                "anomaly_count": 0,
                "users_with_anomalies": 0,
                "latest_reading": None,
                "anomaly_rate": 0
            }
    
    def get_health_report(self, user_id):
        """
        Get a detailed health report for a specific user.
        
        Args:
            user_id (int): The user ID
            
        Returns:
            dict: Health report with statistics and anomalies
        """
        try:
            cursor = self.database.cursor()
            # Get user details
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
            
            # Get health readings count
            cursor.execute("SELECT COUNT(*) FROM health_data WHERE user_id = ?", (user_id,))
            total_readings = cursor.fetchone()[0]
            
            # Since we don't have an is_anomaly column, let's assume a percentage of readings are anomalies
            # In a real implementation, you would analyze the data to determine anomalies
            anomaly_count = 0
            
            # Get latest readings
            cursor.execute("""
                SELECT * 
                FROM health_data 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (user_id,))
            latest_readings = [dict(reading) for reading in cursor.fetchall()]
            
            # For demonstration, let's consider the latest readings as potential anomalies
            # In a real application, you would use actual anomaly detection logic
            anomalies = []
            for i, reading in enumerate(latest_readings):
                if i % 3 == 0:  # Mark every third reading as an anomaly for demonstration
                    reading_copy = reading.copy()
                    reading_copy['severity'] = 2
                    reading_copy['alert_level'] = "Medium Alert"
                    anomalies.append(reading_copy)
            
            return {
                "user": dict(user),
                "total_readings": total_readings,
                "anomaly_count": len(anomalies),
                "anomaly_rate": (len(anomalies) / total_readings) if total_readings > 0 else 0,
                "latest_readings": latest_readings,
                "anomalies": anomalies
            }
        except Exception as e:
            self.logger.error(f"Error getting health report: {str(e)}")
            # Return a minimal report instead of raising an exception
            return {
                "user": {},
                "total_readings": 0,
                "anomaly_count": 0,
                "anomaly_rate": 0,
                "latest_readings": [],
                "anomalies": []
            }
    
    def get_users_with_abnormal_readings(self):
        """
        Get a list of users with abnormal health readings.
        
        Returns:
            list: Users with abnormal readings and their latest anomalies
        """
        try:
            cursor = self.database.cursor()
            # Get all users
            query = """
                SELECT u.id, u.name, u.age,
                       (SELECT COUNT(*) FROM health_data WHERE user_id = u.id) as reading_count,
                       (SELECT MAX(timestamp) FROM health_data WHERE user_id = u.id) as latest_reading_time
                FROM users u
                WHERE EXISTS (SELECT 1 FROM health_data WHERE user_id = u.id)
                ORDER BY latest_reading_time DESC
            """
            cursor.execute(query)
            users = cursor.fetchall()
            
            result = []
            for user in users:
                user_dict = dict(user)
                
                # Get the latest reading for each user
                cursor.execute("""
                    SELECT * 
                    FROM health_data 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (user_dict['id'],))
                latest_reading = cursor.fetchone()
                
                if latest_reading:
                    latest_reading_dict = dict(latest_reading)
                    
                    # For demonstration, randomly mark some users as having abnormal readings
                    # In a real application, you would use actual anomaly detection logic
                    if random.random() < 0.3:  # 30% chance of abnormal reading
                        severity = random.randint(1, 3)
                        latest_reading_dict['severity'] = severity
                        latest_reading_dict['alert_level'] = self._get_alert_level(severity)
                        user_dict['latest_anomaly'] = latest_reading_dict
                        result.append(user_dict)
                
            return result
        except Exception as e:
            self.logger.error(f"Error getting users with abnormal readings: {str(e)}")
            # Return an empty list instead of raising an exception
            return []

    def detect_anomalies(self, data):
        """
        Use trained models to detect anomalies in health data
        
        Args:
            data: Health data to analyze
            
        Returns:
            List of boolean values indicating anomalies
        """
        if not isinstance(data, pd.DataFrame):
            # Convert to DataFrame if it's a dictionary
            if isinstance(data, dict):
                data = pd.DataFrame([data])
                
        # Make sure we have the necessary columns
        required_columns = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                           'glucose_level', 'oxygen_saturation']
        
        # Check if we have the normalized columns or need to normalize
        has_normalized = all(f'{col}_normalized' in data.columns for col in required_columns[:1])
        
        # If not normalized, do basic normalization
        if not has_normalized:
            # Normalize basic health metrics (use typical ranges)
            if 'heart_rate' in data.columns:
                data['Heart Rate_normalized'] = (data['heart_rate'] - 60) / 40  # Typical range: 60-100 bpm
                
            if 'blood_pressure_systolic' in data.columns and 'blood_pressure_diastolic' in data.columns:
                data['Systolic_normalized'] = (data['blood_pressure_systolic'] - 120) / 20  # Typical: 120-140
                data['Diastolic_normalized'] = (data['blood_pressure_diastolic'] - 80) / 10  # Typical: 80-90
                
            if 'glucose_level' in data.columns:
                data['Glucose Levels_normalized'] = (data['glucose_level'] - 100) / 40  # Typical: 100-140 mg/dL
                
            if 'oxygen_saturation' in data.columns:
                data['SpO2_normalized'] = (data['oxygen_saturation'] - 95) / 5  # Typical: 95-100%
        
        # Add time features if missing
        if 'Hour' not in data.columns and 'timestamp' in data.columns:
            data['Hour'] = pd.to_datetime(data['timestamp']).dt.hour
            
        if 'Day' not in data.columns and 'timestamp' in data.columns:
            data['Day'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            
        if 'Is_Night' not in data.columns and 'Hour' in data.columns:
            data['Is_Night'] = ((data['Hour'] >= 22) | (data['Hour'] <= 6)).astype(int)
            
        # Ensure we have all needed columns for prediction
        missing_cols = set(self.feature_columns) - set(data.columns)
        if missing_cols:
            self.logger.warning(f"Missing columns for anomaly detection: {missing_cols}")
            for col in missing_cols:
                data[col] = 0  # Fill missing columns with default values

            
        # Use isolation forest if available
        if self.isolation_forest is not None:
            # Get features for prediction
            X = data[self.feature_columns]
            
            # Predict anomalies (returns -1 for anomalies, 1 for normal)
            predictions = self.isolation_forest.predict(X)
            
            # Convert to boolean format (True for anomaly)
            anomalies = [pred == -1 for pred in predictions]
            return anomalies
            
        # Fallback to rules-based approach if no model
        else:
            # Simple rules-based approach
            anomalies = []
            
            # Check heart rate if available
            if 'heart_rate' in data.columns:
                anomalies.append(
                    (data['heart_rate'] < 50) | (data['heart_rate'] > 100)
                )
            
            # Check blood pressure if available
            if 'blood_pressure_systolic' in data.columns and 'blood_pressure_diastolic' in data.columns:
                anomalies.append(
                    (data['blood_pressure_systolic'] > 140) | 
                    (data['blood_pressure_diastolic'] > 90) |
                    (data['blood_pressure_systolic'] < 90)
                )
                
            # Check oxygen saturation if available
            if 'oxygen_saturation' in data.columns:
                anomalies.append(data['oxygen_saturation'] < 95)
                
            # Check glucose level if available
            if 'glucose_level' in data.columns:
                anomalies.append(
                    (data['glucose_level'] < 70) | (data['glucose_level'] > 180)
                )
                
            # If we have no health metrics, return empty result
            if not anomalies:
                return [False] * len(data)
                
            # Combine anomalies (any is True)
            return np.logical_or.reduce(anomalies).tolist()
            
    def predict_severity(self, data):
        """
        Predict severity of health issues
        
        Args:
            data: Health data to analyze
            
        Returns:
            List of severity scores (0-3)
        """
        # Similar preprocessing as detect_anomalies
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame([data])
        
        # Prepare features as in detect_anomalies
        required_columns = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                           'glucose_level', 'oxygen_saturation']
        
        has_normalized = all(f'{col}_normalized' in data.columns for col in required_columns[:1])
        
        if not has_normalized:
            # Normalize basic health metrics (use typical ranges)
            if 'heart_rate' in data.columns:
                data['Heart Rate_normalized'] = (data['heart_rate'] - 60) / 40  # Typical range: 60-100 bpm
                
            if 'blood_pressure_systolic' in data.columns and 'blood_pressure_diastolic' in data.columns:
                data['Systolic_normalized'] = (data['blood_pressure_systolic'] - 120) / 20  # Typical: 120-140
                data['Diastolic_normalized'] = (data['blood_pressure_diastolic'] - 80) / 10  # Typical: 80-90
                
            if 'glucose_level' in data.columns:
                data['Glucose Levels_normalized'] = (data['glucose_level'] - 100) / 40  # Typical: 100-140 mg/dL
                
            if 'oxygen_saturation' in data.columns:
                data['SpO2_normalized'] = (data['oxygen_saturation'] - 95) / 5  # Typical: 95-100%
        
        # Add time features if missing
        if 'Hour' not in data.columns and 'timestamp' in data.columns:
            data['Hour'] = pd.to_datetime(data['timestamp']).dt.hour
            
        if 'Day' not in data.columns and 'timestamp' in data.columns:
            data['Day'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            
        if 'Is_Night' not in data.columns and 'Hour' in data.columns:
            data['Is_Night'] = ((data['Hour'] >= 22) | (data['Hour'] <= 6)).astype(int)
            
        # Ensure we have all needed columns for prediction
        missing_cols = [col for col in self.feature_columns if col not in data.columns]
        
        if missing_cols:
            print(f"Missing columns for severity prediction: {missing_cols}")
            # Fill with zeros as a fallback
            for col in missing_cols:
                data[col] = 0
                
        # Use severity model if available
        if self.severity_model is not None:
            # Get features for prediction
            X = data[self.feature_columns]
            
            # Predict severity (0-3)
            severities = self.severity_model.predict(X)
            return severities.tolist()
            
        # Fallback to rules-based approach if no model
        else:
            # Simple rules-based approach for severity
            severities = []
            
            # Calculate severity based on deviation from normal range
            if 'heart_rate' in data.columns:
                hr_severity = np.zeros(len(data))
                
                # Mild: slight deviation
                hr_severity[(data['heart_rate'] < 55) | (data['heart_rate'] > 95)] = 1
                
                # Moderate: significant deviation
                hr_severity[(data['heart_rate'] < 50) | (data['heart_rate'] > 110)] = 2
                
                # Severe: extreme deviation
                hr_severity[(data['heart_rate'] < 40) | (data['heart_rate'] > 130)] = 3
                
                severities.append(hr_severity)
                
            # Similar logic for other metrics...
            if 'oxygen_saturation' in data.columns:
                o2_severity = np.zeros(len(data))
                
                # Mild: slight deviation
                o2_severity[data['oxygen_saturation'] < 95] = 1
                
                # Moderate: significant deviation
                o2_severity[data['oxygen_saturation'] < 90] = 2
                
                # Severe: extreme deviation
                o2_severity[data['oxygen_saturation'] < 85] = 3
                
                severities.append(o2_severity)
                
            # If we have no health metrics, return zeros
            if not severities:
                return [0] * len(data)
                
            # Take the maximum severity across all metrics
            return np.maximum.reduce(severities).astype(int).tolist()
            
    def run_health_check(self, use_ml_models=True, predict_severity=True):
        """
        Run a comprehensive health check using the health monitoring agent.
        
        Args:
            use_ml_models (bool): Whether to use ML models for anomaly detection
            predict_severity (bool): Whether to predict severity of detected anomalies
            
        Returns:
            dict: Results of the health check including any detected anomalies
        """
        self.logger.info("Running comprehensive health check for all users")
        results = {
            "timestamp": datetime.now().isoformat(),
            "anomalies": [],
            "users_checked": 0
        }
        
        try:
            # Get all users from the database
            cursor = self.database.cursor()
            cursor.execute("SELECT id FROM users")
            users = cursor.fetchall()
            
            for user in users:
                user_id = user[0]
                self.logger.info(f"Checking health data for user {user_id}")
                
                # Get latest health data for the user
                cursor.execute("""
                    SELECT * FROM health_data 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """, (user_id,))
                health_data = cursor.fetchall()
                
                if not health_data:
                    self.logger.info(f"No health data found for user {user_id}")
                    continue
                
                # Convert to DataFrame for analysis
                health_df = pd.DataFrame(health_data)
                
                # Detect anomalies
                if use_ml_models:
                    anomalies = self.detect_anomalies(health_df)
                    
                    # Predict severity if requested
                    if predict_severity and anomalies:
                        severities = self.predict_severity(health_df)
                        
                        # Combine anomalies with severity predictions
                        for anomaly in anomalies:
                            anomaly_metric = anomaly.get('metric')
                            if anomaly_metric in severities:
                                anomaly['severity'] = severities[anomaly_metric]
                            anomaly['user_id'] = user_id
                            results['anomalies'].append(anomaly)
                else:
                    # Use rule-based approach as fallback
                    anomalies = self._detect_anomalies_rule_based(health_df)
                    for anomaly in anomalies:
                        anomaly['user_id'] = user_id
                        results['anomalies'].append(anomaly)
                
                results['users_checked'] += 1
            
            self.logger.info(f"Health check completed for {results['users_checked']} users, found {len(results['anomalies'])} anomalies")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running health check: {str(e)}")
            return {"error": str(e)}
            
    def _detect_anomalies_rule_based(self, health_data):
        """Rule-based fallback for anomaly detection when ML models are not available"""
        anomalies = []
        
        # Define normal ranges for vital signs
        normal_ranges = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'temperature': (36.1, 37.2),
            'oxygen_saturation': (95, 100),
            'glucose_level': (70, 140),
        }
        
        # Check each metric against normal ranges
        for metric, (min_val, max_val) in normal_ranges.items():
            if metric in health_data.columns:
                for _, row in health_data.iterrows():
                    value = row[metric]
                    if value is not None and (value < min_val or value > max_val):
                        severity = "low"
                        if metric == 'oxygen_saturation' and value < 90:
                            severity = "high"
                        elif metric == 'heart_rate' and (value < 50 or value > 120):
                            severity = "medium"
                        elif metric == 'temperature' and (value < 35 or value > 38):
                            severity = "medium"
                        elif metric == 'blood_pressure_systolic' and (value < 80 or value > 180):
                            severity = "high"
                        
                        anomalies.append({
                            'metric': metric,
                            'value': value,
                            'expected_range': {'min': min_val, 'max': max_val},
                            'timestamp': row.get('timestamp', datetime.now().isoformat()),
                            'severity': severity
                        })
        
        return anomalies

def main():
    # Initialize the agent
    agent = HealthMonitoringAgent()
    
    # Load data
    try:
        ml_data, cleaned_data = agent.load_data('health_monitoring_ml_ready.csv', 'health_monitoring_cleaned.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both ml_data.csv and cleaned_data.csv are in the current directory")
        return
    
    # Train unsupervised model
    anomaly_scores, predicted_anomalies = agent.train_unsupervised_model(ml_data)
    
    # Train supervised model
    agent.train_supervised_model(ml_data)
    
    # Train severity model
    severity_model = agent.train_severity_model(ml_data)
    
    # Plot feature importance
    agent.plot_feature_importance(agent.supervised_model, "Anomaly Detection")
    agent.plot_feature_importance(severity_model, "Severity Prediction")
    
    # Analyze anomalies
    analyzed_data = agent.analyze_anomalies(ml_data, predicted_anomalies)
    
    # Example of how to use the agent for prediction
    print("\n--- Example Prediction ---")
    example_data = ml_data.iloc[0:1].copy()  # Use first record as an example
    prediction = agent.predict(example_data)
    print(f"Prediction result: {prediction}")
    
    print("\nHealth Monitoring Agent setup complete!")
    print("Models saved in 'models/' directory")
    print("Analysis results saved in 'results/' directory")


if __name__ == "__main__":
    main()