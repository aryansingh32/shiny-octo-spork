import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def clean_health_data(file_path):
    """
    Clean and preprocess health monitoring data for ML model training.
    
    Args:
        file_path: Path to the CSV file containing health monitoring data
        
    Returns:
        Cleaned DataFrame ready for ML model training
    """
    print("Loading and cleaning health monitoring data...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Column names in the original dataset: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a sample dataframe based on the visible data in case the file can't be loaded
        columns = ["Device-ID/User-ID", "Timestamp", "Heart Rate", "Heart Rate Below/Above Threshold (Yes/No)", 
                  "Blood Pressure", "Blood Pressure Below/Above Threshold (Yes/No)", "Glucose Levels", 
                  "Glucose Levels Above/Below Threshold (Yes/No)", "SpO2", "SpO2 Below Threshold (Yes/No)", 
                  "Alert Triggered (Yes/No)", "Notification Sent (Yes/No)"]
        df = pd.read_csv(file_path, names=columns)
    
    # Make a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Check for potential spelling variations or case sensitivity in SpO2 column
    oxygen_columns = [col for col in cleaned_df.columns if 'SpO' in col or 'spo' in col.lower() or 'oxygen' in col.lower()]
    if oxygen_columns:
        print(f"Found oxygen-related columns: {oxygen_columns}")
        # Rename to standardized SpO2 if needed
        if 'SpO2' not in cleaned_df.columns and len(oxygen_columns) > 0:
            cleaned_df.rename(columns={oxygen_columns[0]: 'SpO2'}, inplace=True)
    else:
        print("Warning: No SpO2 or oxygen saturation column found in the dataset")
    
    # 1. Convert timestamps to datetime format
    try:
        cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
    except:
        print("Warning: Timestamp conversion issues - using original format")
    
    # 2. Split Blood Pressure into Systolic and Diastolic
    def extract_blood_pressure(bp_string):
        try:
            # Extract the numbers from strings like "136/79 mmHg"
            systolic, diastolic = bp_string.split('/')[0], bp_string.split('/')[1].split(' ')[0]
            return int(systolic), int(diastolic)
        except:
            return np.nan, np.nan
    
    # Apply the function to split blood pressure
    cleaned_df[['Systolic', 'Diastolic']] = pd.DataFrame(
        cleaned_df['Blood Pressure'].apply(extract_blood_pressure).tolist(), 
        index=cleaned_df.index
    )
    
    # 3. Convert Yes/No columns to binary (1/0)
    yes_no_columns = [col for col in cleaned_df.columns if 'Yes/No' in col or 'Threshold' in col 
                      or 'Alert' in col or 'Notification' in col]
    
    for col in yes_no_columns:
        cleaned_df[col] = cleaned_df[col].map({'Yes': 1, 'No': 0})
    
    # Create a simplified column names mapping
    column_mapping = {
        'Heart Rate Below/Above Threshold (Yes/No)': 'Heart_Rate_Alert',
        'Blood Pressure Below/Above Threshold (Yes/No)': 'Blood_Pressure_Alert',
        'Glucose Levels Above/Below Threshold (Yes/No)': 'Glucose_Alert',
        'SpO2 Below Threshold (Yes/No)': 'SpO2_Alert',
        'Alert Triggered (Yes/No)': 'Alert_Triggered',
        'Notification Sent (Yes/No)': 'Notification_Sent',
        'Device-ID/User-ID': 'Device_ID'
    }
    
    # Only rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={old_col: new_col})
    
    # 4. Handle missing values
    # For numeric columns, fill missing values with median
    numeric_cols = ['Heart Rate', 'Systolic', 'Diastolic', 'Glucose Levels']
    
    # Add SpO2 to numeric columns if it exists
    if 'SpO2' in cleaned_df.columns:
        numeric_cols.append('SpO2')
    
    for col in numeric_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)
    
    # 5. Create features for anomaly detection
    # Normalize vital signs for ML model
    scaler = MinMaxScaler()
    
    # Only normalize columns that exist
    for col in numeric_cols:
        if col in cleaned_df.columns:
            col_norm = f"{col}_normalized"
            cleaned_df[col_norm] = scaler.fit_transform(cleaned_df[[col]])
    
    # 6. Create a combined anomaly score (sum of individual alerts)
    alert_cols = [col for col in cleaned_df.columns if 'Alert' in col]
    cleaned_df['Anomaly_Score'] = cleaned_df[alert_cols].sum(axis=1)
    
    # 7. Create time-based features
    if isinstance(cleaned_df['Timestamp'].iloc[0], (datetime, pd.Timestamp)):
        cleaned_df['Hour'] = cleaned_df['Timestamp'].dt.hour
        cleaned_df['Day'] = cleaned_df['Timestamp'].dt.day
        cleaned_df['Is_Night'] = (cleaned_df['Hour'] < 6) | (cleaned_df['Hour'] >= 22)
    
    print("Data cleaning completed.")
    return cleaned_df

def analyze_and_visualize(df):
    """
    Generate basic statistics and visualizations for the cleaned data
    
    Args:
        df: Cleaned DataFrame
    """
    # Identify available vital sign columns
    available_vital_signs = []
    for col in ['Heart Rate', 'Systolic', 'Diastolic', 'Glucose Levels', 'SpO2']:
        if col in df.columns:
            available_vital_signs.append(col)
    
    print("\nBasic Statistics:")
    print(df[available_vital_signs].describe())
    
    print("\nAnomaly Distribution:")
    print(df['Anomaly_Score'].value_counts().sort_index())
    
    print("\nAlert Frequencies:")
    alert_cols = [col for col in df.columns if 'Alert' in col]
    alert_counts = df[alert_cols].sum()
    print(alert_counts)
    
    # Save visualizations to file
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Distribution of vital signs (use only available columns)
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df[available_vital_signs])
    plt.title("Distribution of Vital Signs")
    plt.xticks(rotation=45)
    
    # Plot 2: Alert frequencies
    plt.subplot(2, 2, 2)
    alert_counts.plot(kind='bar')
    plt.title("Alert Frequencies by Type")
    plt.xticks(rotation=45)
    
    # Plot 3: Time of day vs anomalies (if Hour column exists)
    plt.subplot(2, 2, 3)
    if 'Hour' in df.columns:
        sns.scatterplot(x='Hour', y='Anomaly_Score', data=df)
        plt.title("Anomalies by Hour of Day")
    else:
        plt.text(0.5, 0.5, "Time data not available", horizontalalignment='center')
        plt.title("Time Analysis (Not Available)")
    
    # Plot 4: Correlation matrix
    plt.subplot(2, 2, 4)
    vital_cols = available_vital_signs + ['Anomaly_Score']
    sns.heatmap(df[vital_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix of Vital Signs")
    
    plt.tight_layout()
    plt.savefig('health_monitoring_analysis.png')
    print("Visualizations saved to 'health_monitoring_analysis.png'")
    
    return

def prepare_for_ml(df):
    """
    Prepare the final dataset for machine learning
    
    Args:
        df: Cleaned and processed DataFrame
        
    Returns:
        DataFrame ready for ML model training
    """
    # Select relevant features for anomaly detection (only those that exist)
    ml_features = [col for col in df.columns if 'normalized' in col]
    
    # Add time features if they exist
    for col in ['Hour', 'Day', 'Is_Night']:
        if col in df.columns:
            ml_features.append(col)
    
    # Add the target variable (if using supervised learning)
    target = 'Anomaly_Score'
    
    # Create the ML-ready dataframe
    ml_df = df[ml_features + [target]]
    
    # Export to CSV
    ml_df.to_csv('health_monitoring_ml_ready.csv', index=False)
    df.to_csv('health_monitoring_cleaned.csv', index=False)
    
    print("\nML-ready dataset created and saved to 'health_monitoring_ml_ready.csv'")
    print("Full cleaned dataset saved to 'health_monitoring_cleaned.csv'")
    
    return ml_df

def main():
    """
    Main function to run the data cleaning and preparation pipeline
    """
    # File path to the dataset
    file_path = 'health_monitoring.csv'
    
    # Clean the data
    cleaned_df = clean_health_data(file_path)
    
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