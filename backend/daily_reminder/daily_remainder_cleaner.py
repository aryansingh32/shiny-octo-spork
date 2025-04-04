import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def clean_reminder_data(file_path):
    """
    Clean and preprocess daily reminder data for analysis and ML model training.
    
    Args:
        file_path: Path to the CSV file containing daily reminder data
        
    Returns:
        Cleaned DataFrame ready for analysis and ML model training
    """
    print("Loading and cleaning daily reminder data...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Column names in the original dataset: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a dataframe based on the provided sample data
        columns = ["Device-ID/User-ID", "Timestamp", "Reminder Type", "Scheduled Time",
                  "Reminder Sent (Yes/No)", "Acknowledged (Yes/No)"]
        df = pd.read_csv(file_path, names=columns)
    
    # Make a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # 1. Clean column names
    column_mapping = {
        'Device-ID/User-ID': 'Device_ID',
        'Reminder Type': 'Reminder_Type',
        'Scheduled Time': 'Scheduled_Time',
        'Reminder Sent (Yes/No)': 'Reminder_Sent',
        'Acknowledged (Yes/No)': 'Acknowledged'
    }
    
    # Only rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={old_col: new_col})
    
    # Drop any unnamed columns
    unnamed_cols = [col for col in cleaned_df.columns if 'Unnamed' in col]
    if unnamed_cols:
        cleaned_df = cleaned_df.drop(columns=unnamed_cols)
        print(f"Dropped unnamed columns: {unnamed_cols}")
    
    # 2. Convert timestamps to datetime format
    try:
        cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
    except:
        print("Warning: Timestamp conversion issues - using original format")
    
    # 3. Convert scheduled time to datetime format
    if 'Scheduled_Time' in cleaned_df.columns:
        try:
            # Extract date from timestamp and combine with scheduled time
            # First, try to parse time using a specific format to avoid warnings
            try:
                cleaned_df['Scheduled_Time'] = pd.to_datetime(cleaned_df['Scheduled_Time'], format='%H:%M:%S').dt.time
            except:
                # If specific format fails, fall back to more flexible parsing
                cleaned_df['Scheduled_Time'] = pd.to_datetime(cleaned_df['Scheduled_Time']).dt.time
                print("Note: Using flexible time parsing for Scheduled_Time")
            
            # Combine date from timestamp with scheduled time
            cleaned_df['Scheduled_DateTime'] = cleaned_df.apply(
                lambda row: datetime.combine(
                    row['Timestamp'].date() if isinstance(row['Timestamp'], (datetime, pd.Timestamp)) else datetime.now().date(),
                    row['Scheduled_Time']
                ), axis=1
            )
        except Exception as e:
            print(f"Warning: Scheduled Time conversion issues - using original format. Error: {e}")
    
    # 4. Convert Yes/No columns to binary (1/0)
    yes_no_columns = ['Reminder_Sent', 'Acknowledged']
    
    for col in yes_no_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].map({'Yes': 1, 'No': 0})
    
    # 5. Create features for reminder type (one-hot encoding)
    if 'Reminder_Type' in cleaned_df.columns:
        reminder_dummies = pd.get_dummies(cleaned_df['Reminder_Type'], prefix='Type')
        cleaned_df = pd.concat([cleaned_df, reminder_dummies], axis=1)
    
    # 6. Calculate time-based features
    if 'Timestamp' in cleaned_df.columns and 'Scheduled_DateTime' in cleaned_df.columns:
        # Create hour of day feature
        cleaned_df['Hour'] = cleaned_df['Scheduled_DateTime'].dt.hour
        
        # Create day of week feature (0=Monday, 6=Sunday)
        cleaned_df['Day_of_Week'] = cleaned_df['Timestamp'].dt.dayofweek
        
        # Create part of day categories
        conditions = [
            (cleaned_df['Hour'] >= 5) & (cleaned_df['Hour'] < 12),
            (cleaned_df['Hour'] >= 12) & (cleaned_df['Hour'] < 17),
            (cleaned_df['Hour'] >= 17) & (cleaned_df['Hour'] < 22)
        ]
        choices = ['Morning', 'Afternoon', 'Evening']
        cleaned_df['Time_of_Day'] = np.select(conditions, choices, default='Night')
        
        # Create weekend indicator
        cleaned_df['Is_Weekend'] = cleaned_df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate adherence rate
        if 'Reminder_Sent' in cleaned_df.columns and 'Acknowledged' in cleaned_df.columns:
            # Adherence only counts if reminder was sent
            cleaned_df['Adherence'] = np.where(cleaned_df['Reminder_Sent'] == 1, 
                                              cleaned_df['Acknowledged'], 
                                              np.nan)
    
    # 7. Create reminder categories based on time sensitivity
    if 'Reminder_Type' in cleaned_df.columns:
        # Define priority levels for different reminder types
        priority_map = {
            'Medication': 'High',
            'Appointment': 'High',
            'Exercise': 'Medium',
            'Hydration': 'Medium'
        }
        # Map reminder types to priority categories, with 'Low' as default
        cleaned_df['Priority'] = cleaned_df['Reminder_Type'].map(priority_map).fillna('Low')
        
        # Convert priority to numeric
        priority_numeric = {'High': 3, 'Medium': 2, 'Low': 1}
        cleaned_df['Priority_Score'] = cleaned_df['Priority'].map(priority_numeric)
    
    # 8. Calculate compliance metrics
    if 'Reminder_Sent' in cleaned_df.columns and 'Acknowledged' in cleaned_df.columns:
        # Calculate overall compliance
        sent_count = cleaned_df['Reminder_Sent'].sum()
        acknowledged_count = cleaned_df[cleaned_df['Reminder_Sent'] == 1]['Acknowledged'].sum()
        compliance_rate = acknowledged_count / sent_count if sent_count > 0 else 0
        print(f"\nOverall compliance rate: {compliance_rate:.2%}")
        
        # Calculate compliance by reminder type (fixing the deprecation warning)
        if 'Reminder_Type' in cleaned_df.columns:
            # Group by reminder type, then calculate compliance for each type
            reminder_groups = cleaned_df.groupby('Reminder_Type')
            type_compliance = {}
            
            for reminder_type, group in reminder_groups:
                sent = group['Reminder_Sent'].sum()
                acknowledged = group[group['Reminder_Sent'] == 1]['Acknowledged'].sum()
                type_compliance[reminder_type] = acknowledged / sent if sent > 0 else 0
            
            print("\nCompliance rate by reminder type:")
            for reminder_type, rate in type_compliance.items():
                print(f"{reminder_type}: {rate:.6f}")
    
    print("Data cleaning completed.")
    return cleaned_df

def analyze_and_visualize(df):
    """
    Generate basic statistics and visualizations for the cleaned reminder data
    
    Args:
        df: Cleaned DataFrame
    """
    print("\nBasic Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe())
    
    if 'Reminder_Type' in df.columns:
        print("\nReminder Type Distribution:")
        print(df['Reminder_Type'].value_counts())
    
    if 'Time_of_Day' in df.columns:
        print("\nTime of Day Distribution:")
        print(df['Time_of_Day'].value_counts())
    
    # Save visualizations to file
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Reminder type distribution
    plt.subplot(2, 2, 1)
    if 'Reminder_Type' in df.columns:
        sns.countplot(x='Reminder_Type', data=df)
        plt.title("Reminder Type Distribution")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "Reminder type data not available", horizontalalignment='center')
        plt.title("Reminder Type (Not Available)")
    
    # Plot 2: Compliance rate by reminder type (fixing the deprecation warning)
    plt.subplot(2, 2, 2)
    if 'Reminder_Type' in df.columns and 'Acknowledged' in df.columns and 'Reminder_Sent' in df.columns:
        # Calculate compliance manually rather than using apply
        reminder_groups = df.groupby('Reminder_Type')
        compliance_data = []
        
        for reminder_type, group in reminder_groups:
            sent = group['Reminder_Sent'].sum()
            acknowledged = group[group['Reminder_Sent'] == 1]['Acknowledged'].sum()
            compliance_rate = acknowledged / sent if sent > 0 else 0
            compliance_data.append({'Reminder_Type': reminder_type, 'Compliance_Rate': compliance_rate})
        
        compliance_by_type = pd.DataFrame(compliance_data)
        
        sns.barplot(x='Reminder_Type', y='Compliance_Rate', data=compliance_by_type)
        plt.title("Compliance Rate by Reminder Type")
        plt.ylabel("Compliance Rate")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "Compliance data not available", horizontalalignment='center')
        plt.title("Compliance by Type (Not Available)")
    
    # Plot 3: Distribution by hour of day
    plt.subplot(2, 2, 3)
    if 'Hour' in df.columns:
        hour_counts = df['Hour'].value_counts().sort_index()
        sns.barplot(x=hour_counts.index, y=hour_counts.values)
        plt.title("Reminder Distribution by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "Hour data not available", horizontalalignment='center')
        plt.title("Hour Distribution (Not Available)")
    
    # Plot 4: Compliance by time of day (fixing the deprecation warning)
    plt.subplot(2, 2, 4)
    if 'Time_of_Day' in df.columns and 'Acknowledged' in df.columns and 'Reminder_Sent' in df.columns:
        # Calculate compliance manually rather than using apply
        time_groups = df.groupby('Time_of_Day')
        compliance_data = []
        
        for time_of_day, group in time_groups:
            sent = group['Reminder_Sent'].sum()
            acknowledged = group[group['Reminder_Sent'] == 1]['Acknowledged'].sum()
            compliance_rate = acknowledged / sent if sent > 0 else 0
            compliance_data.append({'Time_of_Day': time_of_day, 'Compliance_Rate': compliance_rate})
        
        compliance_by_time = pd.DataFrame(compliance_data)
        
        # Order by time of day
        time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
        compliance_by_time['Time_of_Day'] = pd.Categorical(
            compliance_by_time['Time_of_Day'], 
            categories=time_order, 
            ordered=True
        )
        compliance_by_time = compliance_by_time.sort_values('Time_of_Day')
        
        sns.barplot(x='Time_of_Day', y='Compliance_Rate', data=compliance_by_time)
        plt.title("Compliance Rate by Time of Day")
        plt.ylabel("Compliance Rate")
    else:
        plt.text(0.5, 0.5, "Time of day compliance data not available", horizontalalignment='center')
        plt.title("Compliance by Time (Not Available)")
    
    plt.tight_layout()
    plt.savefig('daily_reminder_analysis.png')
    print("Visualizations saved to 'daily_reminder_analysis.png'")
    
    # Additional visualization - Day of week analysis
    plt.figure(figsize=(10, 6))
    if 'Day_of_Week' in df.columns and 'Acknowledged' in df.columns and 'Reminder_Sent' in df.columns:
        # Calculate compliance by day of week
        day_groups = df.groupby('Day_of_Week')
        compliance_data = []
        
        for day, group in day_groups:
            sent = group['Reminder_Sent'].sum()
            acknowledged = group[group['Reminder_Sent'] == 1]['Acknowledged'].sum()
            compliance_rate = acknowledged / sent if sent > 0 else 0
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(day)]
            compliance_data.append({'Day': day_name, 'Day_Num': day, 'Compliance_Rate': compliance_rate})
        
        compliance_by_day = pd.DataFrame(compliance_data)
        compliance_by_day = compliance_by_day.sort_values('Day_Num')
        
        sns.barplot(x='Day', y='Compliance_Rate', data=compliance_by_day)
        plt.title("Compliance Rate by Day of Week")
        plt.ylabel("Compliance Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('daily_reminder_day_analysis.png')
        print("Day of week analysis saved to 'daily_reminder_day_analysis.png'")
    
    return

def prepare_for_ml(df):
    """
    Prepare the final dataset for machine learning
    
    Args:
        df: Cleaned and processed DataFrame
        
    Returns:
        DataFrame ready for ML model training
    """
    # Select one-hot encoded columns for reminder types
    type_cols = [col for col in df.columns if col.startswith('Type_')]
    
    # Add time-related features
    time_features = []
    for col in ['Hour', 'Day_of_Week', 'Is_Weekend', 'Priority_Score']:
        if col in df.columns:
            time_features.append(col)
    
    # Create the ML-ready dataframe with relevant features
    ml_features = type_cols + time_features
    
    # Add the target variable (Acknowledged status)
    target_variable = []
    if 'Acknowledged' in df.columns:
        target_variable = ['Acknowledged']
    
    # Only include rows where a reminder was sent
    if 'Reminder_Sent' in df.columns:
        ml_df = df[df['Reminder_Sent'] == 1][ml_features + target_variable]
    else:
        ml_df = df[ml_features + target_variable]
    
    # Export to CSV
    ml_df.to_csv('daily_reminder_ml_ready.csv', index=False)
    df.to_csv('daily_reminder_cleaned.csv', index=False)
    
    print("\nML-ready dataset created and saved to 'daily_reminder_ml_ready.csv'")
    print("Full cleaned dataset saved to 'daily_reminder_cleaned.csv'")
    
    # Add feature importance analysis using a simple model
    if len(target_variable) > 0 and len(ml_df) > 0:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            print("\nPerforming initial feature importance analysis...")
            
            X = ml_df.drop(columns=target_variable)
            y = ml_df[target_variable[0]]
            
            # Train a simple model to get feature importance
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Get feature importance
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                print("\nFeature Importance:")
                print(feature_importance)
                
                # Visualize feature importance
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance)
                plt.title("Feature Importance for Reminder Acknowledgment")
                plt.tight_layout()
                plt.savefig('reminder_feature_importance.png')
                print("Feature importance saved to 'reminder_feature_importance.png'")
                
                # Test set performance
                accuracy = model.score(X_test, y_test)
                print(f"Initial model accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"Error in model training: {e}")
        except ImportError:
            print("Note: sklearn not available, skipping feature importance analysis")
    
    return ml_df

def main():
    """
    Main function to run the data cleaning and preparation pipeline
    """
    # File path to the dataset
    file_path = 'daily_reminder.csv'
    
    # Clean the data
    cleaned_df = clean_reminder_data(file_path)
    
    # Analyze and visualize the data
    analyze_and_visualize(cleaned_df)
    
    # Prepare for ML
    ml_df = prepare_for_ml(cleaned_df)
    
    print("\nData preprocessing completed successfully.")
    print(f"Original shape: {cleaned_df.shape}")
    print(f"ML-ready shape: {ml_df.shape}")
    
    # Display a sample of the ML-ready data
    print("\nSample of ML-ready data:")
    print(ml_df.head())

if __name__ == "__main__":
    main()