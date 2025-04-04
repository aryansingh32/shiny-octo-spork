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
import pyttsx3
import logging

# Create directories for models and results
os.makedirs('modelsSafety', exist_ok=True)
os.makedirs('resultsSafety', exist_ok=True)

class SafetyMonitoringAgent:
    def __init__(self, database):
        self.database = database
        self.isolation_forest = None
        self.supervised_model = None
        self.risk_level_model = None
        self.feature_columns = None
        self.threshold = 0.0
        
        # Load pre-trained models if they exist
        self._load_models()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            # Define model file paths
            isolation_forest_path = 'modelsSafety/safety_isolation_forest_model.pkl'
            supervised_model_path = 'modelsSafety/safety_rf_classifier_model.pkl'
            risk_level_model_path = 'modelsSafety/safety_risk_level_model.pkl'
            
            # Check if isolation forest model exists and load it
            if os.path.exists(isolation_forest_path):
                self.isolation_forest = joblib.load(isolation_forest_path)
                print(f"Loaded Safety Isolation Forest model from {isolation_forest_path}")
            else:
                print(f"Safety Isolation Forest model not found at {isolation_forest_path}, will train when data is provided")
                
            # Check if supervised model exists and load it
            if os.path.exists(supervised_model_path):
                self.supervised_model = joblib.load(supervised_model_path)
                print(f"Loaded Safety Supervised model from {supervised_model_path}")
            else:
                print(f"Safety Supervised model not found at {supervised_model_path}, will train when data is provided")
                
            # Check if risk level model exists and load it
            if os.path.exists(risk_level_model_path):
                self.risk_level_model = joblib.load(risk_level_model_path)
                print(f"Loaded Safety Risk Level model from {risk_level_model_path}")
            else:
                print(f"Safety Risk Level model not found at {risk_level_model_path}, will train when data is provided")
                
            # Set default feature columns if not loaded from model
            if self.feature_columns is None:
                # Use default feature columns for safety data
                self.feature_columns = ['Movement_Walking_normalized', 'Movement_Standing_normalized',
                                       'Movement_Sitting_normalized', 'Movement_Lying_normalized',
                                       'Location_Bedroom_normalized', 'Location_Bathroom_normalized',
                                       'Location_Kitchen_normalized', 'Location_LivingRoom_normalized',
                                       'Hour', 'Day', 'Is_Night']
                
        except Exception as e:
            print(f"Error loading pre-trained safety models: {e}")
        
    def load_data(self, ml_data_path, cleaned_data_path=None):
        """Load and prepare the datasets for training and validation"""
        # Load the ML dataset
        ml_data = pd.read_csv(ml_data_path, sep=",")
        
        # Identify normalized feature columns
        normalized_cols = [col for col in ml_data.columns if 'normalized' in col]
        movement_cols = [col for col in ml_data.columns if col.startswith('Movement_')]
        location_cols = [col for col in ml_data.columns if col.startswith('Location_')]
        
        # Time-based features
        time_features = []
        for col in ['Hour', 'Day', 'Is_Night']:
            if col in ml_data.columns:
                time_features.append(col)
                
                # Convert Is_Night to numeric if it's not
                if col == 'Is_Night' and (ml_data[col].dtype == bool or ml_data[col].dtype == object):
                    ml_data[col] = ml_data[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
        
        # Define feature columns for model training
        self.feature_columns = normalized_cols + movement_cols + location_cols + time_features
        
        # Convert all movement columns to numeric (0 or 1)
        for col in movement_cols:
            # Check if column contains non-numeric values
            if ml_data[col].dtype == object:
                # Convert to binary (1 if the column name matches the value, 0 otherwise)
                movement_type = col.replace('Movement_', '')
                ml_data[col] = (ml_data[col] == movement_type).astype(int)
        
        # Similarly check and convert location columns
        for col in location_cols:
            if ml_data[col].dtype == object:
                location_type = col.replace('Location_', '')
                ml_data[col] = (ml_data[col] == location_type).astype(int)
        
        print(f"Loaded ML data with {len(ml_data)} records")
        print(f"Feature columns: {self.feature_columns}")
        
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
        X = data[self.feature_columns].copy()
        
        # Make sure all data is numeric
        for col in X.columns:
            if X[col].dtype == object:
                print(f"Converting column {col} to numeric")
                # Try to convert to numeric, set non-convertible values to 0
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
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
        joblib.dump(self.isolation_forest, 'modelsSafety/safety_isolation_forest_model.pkl')
        print("Isolation Forest model trained and saved!")
        
        return anomaly_scores, predicted_anomalies
    
    def train_supervised_model(self, data):
        """Train a supervised model for fall detection and risk assessment"""
        print("\n--- Training Supervised Classification Model ---")
        
        # Prepare features
        X = data[self.feature_columns].copy()
        
        # Ensure all X data is numeric
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Check if Fall_Detected exists, otherwise use Risk_Score
        if 'Fall_Detected' in data.columns:
            y = data['Fall_Detected']
            target_name = 'Fall_Detected'
        else:
            # If no Fall_Detected column, use Risk_Score thresholded to binary
            y = (data['Risk_Score'] > 0).astype(int)
            target_name = 'Risk_Binary'
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
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
        print(f"\nSupervised Model Classification Report for {target_name}:")
        print(classification_report(y_test, y_pred))
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {target_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'resultsSafety/safety_confusion_matrix_{target_name.lower()}.png')
        
        # Save the model
        joblib.dump(self.supervised_model, 'modelsSafety/safety_rf_classifier_model.pkl')
        print(f"Random Forest classifier for {target_name} trained and saved!")
        
        return y_pred
    
    def train_risk_level_model(self, data):
        """Train a model to predict the risk level (0-3)"""
        print("\n--- Training Risk Level Model ---")
        
        # Prepare features
        X = data[self.feature_columns].copy()
        
        # Ensure all X data is numeric
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # If Risk_Score exists, use it, otherwise create from Fall_Detected
        if 'Risk_Score' in data.columns:
            # Convert continuous risk score to discrete levels (0-3)
            risk_max = data['Risk_Score'].max()
            # Ensure risk_max is not zero to avoid division by zero
            if risk_max == 0:
                risk_max = 1  
            # Use pd.cut but handle NaN values before converting to int
            y_cut = pd.cut(
                data['Risk_Score'], 
                bins=[0, risk_max/4, risk_max/2, 3*risk_max/4, float('inf')],
                labels=[0, 1, 2, 3]
            )
            # Fill NaN values with 0 before converting to int
            y = y_cut.fillna(0).astype(int)
        else:
            # If no Risk_Score, use Fall_Detected as binary risk
            y = data['Fall_Detected']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest classifier for risk levels
        risk_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        risk_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = risk_model.predict(X_test)
        print("\nRisk Level Model Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        joblib.dump(risk_model, 'modelsSafety/safety_risk_level_model.pkl')
        print("Risk level prediction model trained and saved!")
        
        return risk_model
    
    def plot_feature_importance(self, model, title):
        """Plot feature importance for a given model"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Limit to top 15 features if there are many
        if len(importance) > 15:
            indices = indices[:15]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [self.feature_columns[i] for i in indices], rotation=45)
        plt.title(f'Feature Importance: {title}')
        plt.tight_layout()
        plt.savefig(f'resultsSafety/safety_feature_importance_{title.lower().replace(" ", "_")}.png')
    
    def analyze_safety_patterns(self, data, predictions):
        """Analyze detected safety issues and their characteristics"""
        # Add predictions to the dataframe
        data_with_preds = data.copy()
        data_with_preds['Predicted_Risk'] = predictions
        
        # Compare predicted vs actual risk events
        print("\n--- Safety Pattern Analysis ---")
        
        # Analyze risk distribution
        if 'Risk_Score' in data_with_preds.columns:
            print(f"Risk score distribution:\n{data_with_preds['Risk_Score'].describe()}")
            
            # Visualize risk score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data_with_preds['Risk_Score'], kde=True)
            plt.title('Distribution of Risk Scores')
            plt.tight_layout()
            plt.savefig('resultsSafety/safety_risk_distribution.png')
        
        # Analyze patterns by location
        location_cols = [col for col in data_with_preds.columns if col.startswith('Location_')]
        if location_cols and 'Risk_Score' in data_with_preds.columns:
            location_risk = pd.DataFrame()
            for loc in location_cols:
                location_name = loc.replace('Location_', '')
                # Calculate average risk for each location
                location_data = data_with_preds[data_with_preds[loc] == 1]
                if not location_data.empty:
                    avg_risk = location_data['Risk_Score'].mean()
                    # Use concat instead of append (which is deprecated)
                    new_row = pd.DataFrame({'Location': [location_name], 'Average_Risk': [avg_risk]})
                    location_risk = pd.concat([location_risk, new_row], ignore_index=True)
            
            if not location_risk.empty:
                location_risk = location_risk.sort_values(by='Average_Risk', ascending=False)
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Location', y='Average_Risk', data=location_risk)
                plt.title('Average Risk Score by Location')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('resultsSafety/safety_risk_by_location.png')
        
        # Analyze time patterns
        if 'Hour' in data_with_preds.columns and 'Risk_Score' in data_with_preds.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='Hour', y='Risk_Score', data=data_with_preds)
            plt.title('Risk Score by Hour of Day')
            plt.tight_layout()
            plt.savefig('resultsSafety/safety_risk_by_hour.png')
        
        # Analyze by movement activity
        movement_cols = [col for col in data_with_preds.columns if col.startswith('Movement_')]
        if movement_cols and 'Risk_Score' in data_with_preds.columns:
            movement_risk = pd.DataFrame()
            for mov in movement_cols:
                movement_name = mov.replace('Movement_', '')
                # Calculate average risk for each movement type
                movement_data = data_with_preds[data_with_preds[mov] == 1]
                if not movement_data.empty:
                    avg_risk = movement_data['Risk_Score'].mean()
                    # Use concat instead of append
                    new_row = pd.DataFrame({'Movement': [movement_name], 'Average_Risk': [avg_risk]})
                    movement_risk = pd.concat([movement_risk, new_row], ignore_index=True)
            
            if not movement_risk.empty:
                movement_risk = movement_risk.sort_values(by='Average_Risk', ascending=False)
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Movement', y='Average_Risk', data=movement_risk)
                plt.title('Average Risk Score by Movement Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('resultsSafety/safety_risk_by_movement.png')
        
        # Save analysis results
        data_with_preds.to_csv('resultsSafety/safety_analyzed_data.csv', index=False)
        
        return data_with_preds
    
    def predict(self, new_data):
        """Make predictions on new safety monitoring data and generate alerts."""
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
        
        # Ensure all data is numeric
        X = new_data[self.feature_columns].copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Make predictions
        is_risk = self.supervised_model.predict(X)
        
        # Get risk level if a risk is detected
        risk_level = 0
        if is_risk[0] == 1:
            risk_model = joblib.load('modelsSafety/safety_risk_level_model.pkl')
            risk_level = risk_model.predict(X)[0]
        
        result = {
            'is_risk_detected': bool(is_risk[0]),
            'risk_level': int(risk_level),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Get explanation for the prediction
        if result['is_risk_detected']:
            result['explanation'] = self._get_risk_explanation(new_data)
            result['recommended_action'] = self._get_recommended_action(risk_level)
        
        return result
    
    def _get_risk_explanation(self, data):
        """Generate an explanation for why this is flagged as a risk."""
        # Get feature importances
        importances = self.supervised_model.feature_importances_
        feature_importance_dict = dict(zip(self.feature_columns, importances))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top 3 most important features
        top_features = sorted_features[:3]
        
        # Prepare explanation
        explanation = "Risk detected due to: "
        
        for feature, _ in top_features:
            # Format explanation based on feature type
            if feature in data.columns:
                if feature.startswith('Movement_'):
                    movement_type = feature.replace('Movement_', '')
                    explanation += f"unusual movement '{movement_type}', "
                elif feature.startswith('Location_'):
                    location = feature.replace('Location_', '')
                    explanation += f"being in '{location}', "
                elif 'Impact_Force' in feature and pd.to_numeric(data[feature].iloc[0], errors='coerce') > 0.5:
                    explanation += f"high impact force, "
                elif 'Inactivity_Duration' in feature and pd.to_numeric(data[feature].iloc[0], errors='coerce') > 0.5:
                    explanation += f"extended inactivity, "
                elif feature == 'Is_Night' and pd.to_numeric(data[feature].iloc[0], errors='coerce') == 1:
                    explanation += f"activity during nighttime, "
        
        return explanation.rstrip(", ")
    
    def _get_recommended_action(self, risk_level):
        """Determine recommended actions based on risk level."""
        if risk_level == 0:
            return "No action needed"
        elif risk_level == 1:
            return "Monitoring suggested. No immediate action required"
        elif risk_level == 2:
            return "Check on the individual. Verbal confirmation of status recommended"
        else:  # risk_level == 3
            return "IMMEDIATE ACTION REQUIRED. Potential fall or dangerous situation detected"
    
    def _get_alert_level(self, risk_level):
        """Determine alert level based on risk level"""
        if risk_level < 0.3:
            return "Normal"
        elif risk_level < 0.6:
            return "Low Alert"
        elif risk_level < 0.8:
            return "Medium Alert"
        else:  # risk_level >= 0.8
            return "High Alert - Immediate Attention Required"
    
    def get_alerts(self):
        """
        Retrieve safety alerts from the database.
        
        Returns:
            list: A list of safety alerts with their details
        """
        try:
            cursor = self.database.cursor()
            query = """
                SELECT s.*, u.name as user_name 
                FROM safety_data s
                JOIN users u ON s.user_id = u.id
                WHERE s.risk_level > 0.5 OR s.fall_detected = 1
                ORDER BY s.timestamp DESC
                LIMIT 100
            """
            cursor.execute(query)
            alerts = cursor.fetchall()
            
            # Convert to list of dictionaries
            result = []
            for alert in alerts:
                alert_dict = dict(alert)
                # Add the alert level based on risk level
                if 'risk_level' in alert_dict:
                    alert_dict['alert_level'] = self._get_alert_level(alert_dict['risk_level'])
                result.append(alert_dict)
                
            return result
        except Exception as e:
            self.logger.error(f"Error getting safety alerts: {str(e)}")
            raise

    def detect_falls(self, data):
        """
        Use trained models to detect falls in safety data
        
        Args:
            data: Safety data to analyze
            
        Returns:
            List of boolean values indicating detected falls
        """
        if not isinstance(data, pd.DataFrame):
            # Convert to DataFrame if it's a dictionary
            if isinstance(data, dict):
                data = pd.DataFrame([data])
                
        # Check if data already contains fall detection
        if 'fall_detected' in data.columns:
            return data['fall_detected'].astype(bool).tolist()
                
        # Make sure we have feature columns defined
        if self.feature_columns is None:
            self.feature_columns = ['Movement_Walking_normalized', 'Movement_Standing_normalized',
                                  'Movement_Sitting_normalized', 'Movement_Lying_normalized',
                                  'Location_Bedroom_normalized', 'Location_Bathroom_normalized',
                                  'Location_Kitchen_normalized', 'Location_LivingRoom_normalized',
                                  'Hour', 'Day', 'Is_Night']
                
        # Normalize movement data if needed
        movement_types = ['Walking', 'Standing', 'Sitting', 'Lying']
        for move_type in movement_types:
            col = f'Movement_{move_type}'
            norm_col = f'{col}_normalized'
            
            if norm_col not in data.columns and col in data.columns:
                # Convert category to binary (1 if this movement, 0 otherwise)
                if data[col].dtype == object:
                    data[norm_col] = (data[col] == move_type).astype(int)
                else:
                    data[norm_col] = data[col]
                    
        # Normalize location data if needed
        location_types = ['Bedroom', 'Bathroom', 'Kitchen', 'LivingRoom']
        for loc_type in location_types:
            col = f'Location_{loc_type}'
            norm_col = f'{col}_normalized'
            
            if norm_col not in data.columns and 'location' in data.columns:
                # Convert category to binary (1 if this location, 0 otherwise)
                data[norm_col] = (data['location'] == loc_type).astype(int)
                    
        # Add time features if missing
        if 'Hour' not in data.columns and 'timestamp' in data.columns:
            data['Hour'] = pd.to_datetime(data['timestamp']).dt.hour
            
        if 'Day' not in data.columns and 'timestamp' in data.columns:
            data['Day'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            
        if 'Is_Night' not in data.columns and 'Hour' in data.columns:
            data['Is_Night'] = ((data['Hour'] >= 22) | (data['Hour'] <= 6)).astype(int)
            
        # Ensure we have all needed columns for prediction
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0  # Default value
            
        # Use fall detection model if available
        if self.supervised_model is not None:
            # Get features for prediction
            X = data[self.feature_columns]
            
            # Predict falls
            predictions = self.supervised_model.predict(X)
            return predictions.astype(bool).tolist()
            
        # Fallback to rules-based approach if no model
        else:
            # Simple rules-based approach for fall detection
            falls = []
            
            # Rule 1: Check if movement transitioned from active to inactive suddenly
            if 'movement_type' in data.columns and len(data) > 1:
                # Get movement levels (from most active to least)
                movement_levels = {
                    'Walking': 3,
                    'Standing': 2,
                    'Sitting': 1,
                    'Lying': 0
                }
                
                # Convert to numeric levels
                if data['movement_type'].dtype == object:
                    movement_values = data['movement_type'].map(lambda x: movement_levels.get(x, 0))
                    
                    # Detect sudden drops in movement level
                    drops = movement_values.diff() <= -2  # Drop of 2 or more levels
                    falls.append(drops)
            
            # Rule 2: Check location transitions (bathroom falls common)
            if 'location' in data.columns and len(data) > 1:
                # Check for transition to bathroom with movement drop
                if 'movement_type' in data.columns:
                    bathroom_transition = (data['location'] == 'Bathroom') & (data['movement_type'] == 'Lying')
                    falls.append(bathroom_transition)
            
            # If we have no safety metrics, return empty result
            if not falls:
                return [False] * len(data)
                
            # Combine fall detections (any is True)
            return np.logical_or.reduce(falls).fillna(False).tolist()
            
    def predict_risk_level(self, data):
        """
        Predict risk level for safety issues
        
        Args:
            data: Safety data to analyze
            
        Returns:
            List of risk level scores (0-3)
        """
        if not isinstance(data, pd.DataFrame):
            # Convert to DataFrame if it's a dictionary
            if isinstance(data, dict):
                data = pd.DataFrame([data])
                
        # Check if data already contains risk score
        if 'risk_score' in data.columns:
            # Convert continuous risk score to discrete levels (0-3)
            risk_max = data['risk_score'].max()
            # Ensure risk_max is not zero to avoid division by zero
            if risk_max == 0:
                risk_max = 1  
            # Use pd.cut but handle NaN values before converting to int
            y_cut = pd.cut(
                data['risk_score'], 
                bins=[0, risk_max/4, risk_max/2, 3*risk_max/4, float('inf')],
                labels=[0, 1, 2, 3]
            )
            # Fill NaN values with 0 before converting to int
            return y_cut.fillna(0).astype(int).tolist()
                
        # Make sure we have feature columns defined (same as detect_falls)
        if self.feature_columns is None:
            self.feature_columns = ['Movement_Walking_normalized', 'Movement_Standing_normalized',
                                  'Movement_Sitting_normalized', 'Movement_Lying_normalized',
                                  'Location_Bedroom_normalized', 'Location_Bathroom_normalized',
                                  'Location_Kitchen_normalized', 'Location_LivingRoom_normalized',
                                  'Hour', 'Day', 'Is_Night']
        
        # Prepare data similarly to detect_falls method
        # (This would duplicate the preprocessing code from detect_falls)
        # Normalize movement data if needed
        movement_types = ['Walking', 'Standing', 'Sitting', 'Lying']
        for move_type in movement_types:
            col = f'Movement_{move_type}'
            norm_col = f'{col}_normalized'
            
            if norm_col not in data.columns and col in data.columns:
                # Convert category to binary (1 if this movement, 0 otherwise)
                if data[col].dtype == object:
                    data[norm_col] = (data[col] == move_type).astype(int)
                else:
                    data[norm_col] = data[col]
                    
        # Normalize location data if needed
        location_types = ['Bedroom', 'Bathroom', 'Kitchen', 'LivingRoom']
        for loc_type in location_types:
            col = f'Location_{loc_type}'
            norm_col = f'{col}_normalized'
            
            if norm_col not in data.columns and 'location' in data.columns:
                # Convert category to binary (1 if this location, 0 otherwise)
                data[norm_col] = (data['location'] == loc_type).astype(int)
                    
        # Add time features if missing
        if 'Hour' not in data.columns and 'timestamp' in data.columns:
            data['Hour'] = pd.to_datetime(data['timestamp']).dt.hour
            
        if 'Day' not in data.columns and 'timestamp' in data.columns:
            data['Day'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            
        if 'Is_Night' not in data.columns and 'Hour' in data.columns:
            data['Is_Night'] = ((data['Hour'] >= 22) | (data['Hour'] <= 6)).astype(int)
            
        # Ensure we have all needed columns for prediction
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0  # Default value
            
        # Use risk level model if available
        if self.risk_level_model is not None:
            # Get features for prediction
            X = data[self.feature_columns]
            
            # Predict risk levels (0-3)
            risk_levels = self.risk_level_model.predict(X)
            return risk_levels.astype(int).tolist()
            
        # Fallback to rules-based approach if no model
        else:
            # Simple rules-based approach for risk assessment
            
            # Initialize risk levels at 0
            risk_levels = np.zeros(len(data))
            
            # Rule 1: Higher risk at night
            if 'Hour' in data.columns:
                night_hours = (data['Hour'] >= 22) | (data['Hour'] <= 6)
                risk_levels[night_hours] += 1
            
            # Rule 2: Higher risk in bathroom
            if 'location' in data.columns:
                bathroom_location = data['location'] == 'Bathroom'
                risk_levels[bathroom_location] += 1
            
            # Rule 3: Lying down in unusual places
            if 'movement_type' in data.columns and 'location' in data.columns:
                unusual_lying = (data['movement_type'] == 'Lying') & ~(data['location'] == 'Bedroom')
                risk_levels[unusual_lying] += 2
            
            # Rule 4: Inactivity for long periods
            if 'movement_type' in data.columns and len(data) > 1:
                # Convert to numeric levels
                movement_levels = {
                    'Walking': 3,
                    'Standing': 2,
                    'Sitting': 1,
                    'Lying': 0
                }
                
                if data['movement_type'].dtype == object:
                    data['movement_level'] = data['movement_type'].map(lambda x: movement_levels.get(x, 0))
                    
                    # Check for prolonged inactivity (no movement change with low level)
                    if len(data) > 5:  # Need enough data points
                        for i in range(5, len(data)):
                            window = data['movement_level'].iloc[i-5:i]
                            # If consistently low activity for 5 timestamps
                            if (window <= 1).all():
                                risk_levels[i] += 1
            
            # Cap risk levels at 3
            risk_levels = np.minimum(risk_levels, 3)
            
            return risk_levels.astype(int).tolist()

    def monitor_safety(self, use_ml_detection=True, predict_risk=True):
        """
        Monitor safety for all users using the safety monitoring agent.
        
        Args:
            use_ml_detection (bool): Whether to use ML models for fall detection
            predict_risk (bool): Whether to predict risk levels
            
        Returns:
            dict: Results of the safety monitoring including any detected issues
        """
        self.logger.info("Running safety monitoring for all users")
        results = {
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "users_checked": 0
        }
        
        try:
            # Get all users from the database
            cursor = self.database.cursor()
            cursor.execute("SELECT id FROM users")
            users = cursor.fetchall()
            
            for user in users:
                user_id = user[0]
                self.logger.info(f"Checking safety data for user {user_id}")
                
                # Get latest safety data for the user
                cursor.execute("""
                    SELECT * FROM safety_data 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """, (user_id,))
                safety_data = cursor.fetchall()
                
                if not safety_data:
                    self.logger.info(f"No safety data found for user {user_id}")
                    continue
                
                # Convert to DataFrame for analysis
                safety_df = pd.DataFrame(safety_data)
                
                # Detect falls
                if use_ml_detection:
                    falls = self.detect_falls(safety_df)
                    
                    # Process detected falls
                    if falls:
                        for fall in falls:
                            fall['user_id'] = user_id
                            fall['issue_type'] = 'fall'
                            results['issues'].append(fall)
                else:
                    # Use rule-based approach as fallback
                    falls = self._detect_falls_rule_based(safety_df)
                    for fall in falls:
                        fall['user_id'] = user_id
                        fall['issue_type'] = 'fall'
                        results['issues'].append(fall)
                
                # Predict risk levels
                if predict_risk:
                    risk_levels = self.predict_risk_level(safety_df)
                    
                    # Add high risk situations to issues
                    for risk in risk_levels:
                        if risk.get('risk_level') == 'high':
                            risk['user_id'] = user_id
                            risk['issue_type'] = 'high_risk'
                            results['issues'].append(risk)
                
                results['users_checked'] += 1
            
            self.logger.info(f"Safety monitoring completed for {results['users_checked']} users, found {len(results['issues'])} issues")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running safety monitoring: {str(e)}")
            return {"error": str(e)}
            
    def _detect_falls_rule_based(self, safety_data):
        """Rule-based fallback for fall detection when ML models are not available"""
        falls = []
        
        # Check for fall_detected flag in data
        if 'fall_detected' in safety_data.columns:
            for _, row in safety_data.iterrows():
                if row['fall_detected'] == 1 or row['fall_detected'] is True:
                    falls.append({
                        'timestamp': row.get('timestamp', datetime.now().isoformat()),
                        'location': row.get('location', 'unknown'),
                        'severity': 'high',
                        'confidence': 1.0,
                        'details': 'Fall detected by sensor'
                    })
        
        # Check for sudden changes in movement or activity
        if all(col in safety_data.columns for col in ['movement_type', 'activity_level']):
            prev_movement = None
            prev_activity = None
            
            for _, row in sorted(safety_data.iterrows(), key=lambda x: x[1].get('timestamp', '')):
                curr_movement = row.get('movement_type')
                curr_activity = row.get('activity_level')
                
                # Detect potential falls based on sudden activity changes
                if prev_activity in ['moderate', 'high'] and curr_activity == 'inactive' and curr_movement == 'lying':
                    falls.append({
                        'timestamp': row.get('timestamp', datetime.now().isoformat()),
                        'location': row.get('location', 'unknown'),
                        'severity': 'medium',
                        'confidence': 0.7,
                        'details': 'Potential fall detected based on sudden activity change'
                    })
                
                prev_movement = curr_movement
                prev_activity = curr_activity
        
        return falls


def main():
    # Initialize the agent
    agent = SafetyMonitoringAgent()
    
    # Load data
    try:
        ml_data, cleaned_data = agent.load_data('safety_monitoring_ml_ready.csv', 'safety_monitoring_cleaned.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both safety_monitoring_ml_ready.csv and safety_monitoring_cleaned.csv are in the current directory")
        return
    
    # Train unsupervised model
    anomaly_scores, predicted_anomalies = agent.train_unsupervised_model(ml_data)
    
    # Train supervised model
    agent.train_supervised_model(ml_data)
    
    # Train risk level model
    risk_model = agent.train_risk_level_model(ml_data)
    
    # Plot feature importance
    agent.plot_feature_importance(agent.supervised_model, "Risk Detection")
    agent.plot_feature_importance(risk_model, "Risk Level Prediction")
    
    # Analyze safety patterns
    analyzed_data = agent.analyze_safety_patterns(ml_data, predicted_anomalies)
    
    # Example of how to use the agent for prediction
    print("\n--- Example Prediction ---")
    example_data = ml_data.iloc[0:1].copy()  # Use first record as an example
    prediction = agent.predict(example_data)
    print(f"Prediction result: {prediction}")
    
    print("\nSafety Monitoring Agent setup complete!")
    print("Models saved in 'modelsSafety/' directory")
    print("Analysis results saved in 'resultsSafety/' directory")


if __name__ == "__main__":
    main()