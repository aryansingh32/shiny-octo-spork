import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def clean_safety_data(file_path):
    """
    Clean and preprocess safety monitoring data for ML model training.
    
    Args:
        file_path: Path to the CSV file containing safety monitoring data
        
    Returns:
        Cleaned DataFrame ready for ML model training
    """
    print("Loading and cleaning safety monitoring data...")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"Column names in the original dataset: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a dataframe based on the provided sample data
        columns = ["Device-ID/User-ID", "Timestamp", "Movement Activity", "Fall Detected (Yes/No)", 
                  "Impact Force Level", "Post-Fall Inactivity Duration (Seconds)", "Location", 
                  "Alert Triggered (Yes/No)", "Caregiver Notified (Yes/No)"]
        df = pd.read_csv(file_path, names=columns)
    
    # Make a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # 1. Convert timestamps to datetime format
    try:
        cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
    except:
        print("Warning: Timestamp conversion issues - using original format")
    
    # 2. Convert Yes/No columns to binary (1/0)
    yes_no_columns = [col for col in cleaned_df.columns if 'Yes/No' in col 
                      or 'Alert' in col or 'Notified' in col or 'Detected' in col]
    
    for col in yes_no_columns:
        cleaned_df[col] = cleaned_df[col].map({'Yes': 1, 'No': 0})
    
    # 3. Create a simplified column names mapping
    column_mapping = {
        'Fall Detected (Yes/No)': 'Fall_Detected',
        'Alert Triggered (Yes/No)': 'Alert_Triggered',
        'Caregiver Notified (Yes/No)': 'Caregiver_Notified',
        'Device-ID/User-ID': 'Device_ID',
        'Movement Activity': 'Movement_Activity',
        'Impact Force Level': 'Impact_Force',
        'Post-Fall Inactivity Duration (Seconds)': 'Inactivity_Duration'
    }
    
    # Only rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={old_col: new_col})
    
    # 4. Handle missing values and convert impact force
    # Convert impact force from categorical or dash to numeric
    if 'Impact_Force' in cleaned_df.columns:
        # Replace dashes or non-numeric values with NaN
        cleaned_df['Impact_Force'] = pd.to_numeric(cleaned_df['Impact_Force'].replace('-', np.nan), errors='coerce')
        # Fill with 0 for non-fall events
        cleaned_df['Impact_Force'] = cleaned_df['Impact_Force'].fillna(0)
    
    # For inactivity duration, assume 0 for missing values (non-fall events)
    if 'Inactivity_Duration' in cleaned_df.columns:
        cleaned_df['Inactivity_Duration'] = pd.to_numeric(cleaned_df['Inactivity_Duration'], errors='coerce')
        cleaned_df['Inactivity_Duration'] = cleaned_df['Inactivity_Duration'].fillna(0)
    
    # 5. Create features for movement activity (one-hot encoding)
    if 'Movement_Activity' in cleaned_df.columns:
        # One-hot encode the movement activity
        movement_dummies = pd.get_dummies(cleaned_df['Movement_Activity'], prefix='Movement')
        cleaned_df = pd.concat([cleaned_df, movement_dummies], axis=1)
    
    # 6. Create features for location (one-hot encoding)
    if 'Location' in cleaned_df.columns:
        # One-hot encode the location
        location_dummies = pd.get_dummies(cleaned_df['Location'], prefix='Location')
        cleaned_df = pd.concat([cleaned_df, location_dummies], axis=1)
    
    # 7. Normalize numeric features for ML model
    numeric_cols = []
    if 'Impact_Force' in cleaned_df.columns:
        numeric_cols.append('Impact_Force')
    if 'Inactivity_Duration' in cleaned_df.columns:
        numeric_cols.append('Inactivity_Duration')
    
    scaler = MinMaxScaler()
    for col in numeric_cols:
        if col in cleaned_df.columns and cleaned_df[col].max() > 0:  # Only scale if there are non-zero values
            col_norm = f"{col}_normalized"
            cleaned_df[col_norm] = scaler.fit_transform(cleaned_df[[col]])
    
    # 8. Create a combined risk score (based on fall detection, impact force, and inactivity duration)
    cleaned_df['Risk_Score'] = 0
    if 'Fall_Detected' in cleaned_df.columns:
        cleaned_df['Risk_Score'] += cleaned_df['Fall_Detected'] * 5  # Higher weight for fall detection
    
    if 'Impact_Force' in cleaned_df.columns and cleaned_df['Impact_Force'].max() > 0:
        # Normalize impact force to 0-1 range and scale to contribute up to 3 points
        impact_score = cleaned_df['Impact_Force'] / cleaned_df['Impact_Force'].max() * 3
        cleaned_df['Risk_Score'] += impact_score
    
    if 'Inactivity_Duration' in cleaned_df.columns and cleaned_df['Inactivity_Duration'].max() > 0:
        # Normalize inactivity duration to 0-1 range and scale to contribute up to 2 points
        inactivity_score = cleaned_df['Inactivity_Duration'] / cleaned_df['Inactivity_Duration'].max() * 2
        cleaned_df['Risk_Score'] += inactivity_score
    
    # 9. Create time-based features
    if isinstance(cleaned_df['Timestamp'].iloc[0], (datetime, pd.Timestamp)):
        cleaned_df['Hour'] = cleaned_df['Timestamp'].dt.hour
        cleaned_df['Day'] = cleaned_df['Timestamp'].dt.day
        cleaned_df['Is_Night'] = (cleaned_df['Hour'] < 6) | (cleaned_df['Hour'] >= 22)
    
    print("Data cleaning completed.")
    return cleaned_df

def analyze_and_visualize(df):
    """
    Generate basic statistics and visualizations for the cleaned safety data
    
    Args:
        df: Cleaned DataFrame
    """
    # Identify available metrics columns
    available_metrics = []
    for col in ['Impact_Force', 'Inactivity_Duration', 'Risk_Score']:
        if col in df.columns:
            available_metrics.append(col)
    
    print("\nBasic Statistics:")
    print(df[available_metrics].describe())
    
    print("\nFall Detection Summary:")
    if 'Fall_Detected' in df.columns:
        print(f"Total Fall Events: {df['Fall_Detected'].sum()}")
        print(f"Percentage of Falls: {df['Fall_Detected'].mean() * 100:.2f}%")
    
    print("\nLocation Distribution:")
    if 'Location' in df.columns:
        print(df['Location'].value_counts())
    
    print("\nMovement Activity Distribution:")
    if 'Movement_Activity' in df.columns:
        print(df['Movement_Activity'].value_counts())
    
    # Save visualizations to file
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Distribution of metrics
    plt.subplot(2, 2, 1)
    if available_metrics:
        sns.boxplot(data=df[available_metrics])
        plt.title("Distribution of Safety Metrics")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "No metric data available", horizontalalignment='center')
        plt.title("Safety Metrics (Not Available)")
    
    # Plot 2: Movement activity distribution
    plt.subplot(2, 2, 2)
    if 'Movement_Activity' in df.columns:
        movement_counts = df['Movement_Activity'].value_counts()
        movement_counts.plot(kind='bar')
        plt.title("Movement Activity Distribution")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "Movement activity data not available", horizontalalignment='center')
        plt.title("Movement Activity (Not Available)")
    
    # Plot 3: Location distribution
    plt.subplot(2, 2, 3)
    if 'Location' in df.columns:
        location_counts = df['Location'].value_counts()
        location_counts.plot(kind='bar')
        plt.title("Location Distribution")
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "Location data not available", horizontalalignment='center')
        plt.title("Location Distribution (Not Available)")
    
    # Plot 4: Time of day analysis
    plt.subplot(2, 2, 4)
    if 'Hour' in df.columns and 'Risk_Score' in df.columns:
        sns.lineplot(x='Hour', y='Risk_Score', data=df)
        plt.title("Risk Score by Hour of Day")
    else:
        plt.text(0.5, 0.5, "Time or risk score data not available", horizontalalignment='center')
        plt.title("Time Analysis (Not Available)")
    
    plt.tight_layout()
    plt.savefig('safety_monitoring_analysis.png')
    print("Visualizations saved to 'safety_monitoring_analysis.png'")
    
    return

def prepare_for_ml(df):
    """
    Prepare the final dataset for machine learning
    
    Args:
        df: Cleaned and processed DataFrame
        
    Returns:
        DataFrame ready for ML model training
    """
    # Select normalized features for ML
    normalized_features = [col for col in df.columns if 'normalized' in col]
    
    # Select one-hot encoded columns for movement and location
    movement_cols = [col for col in df.columns if col.startswith('Movement_')]
    location_cols = [col for col in df.columns if col.startswith('Location_')]
    
    # Add time features if they exist
    time_features = []
    for col in ['Hour', 'Day', 'Is_Night']:
        if col in df.columns:
            time_features.append(col)
    
    # Combine all features
    ml_features = normalized_features + movement_cols + location_cols + time_features
    
    # Add the target variables (if using supervised learning)
    target_variables = []
    for col in ['Fall_Detected', 'Alert_Triggered', 'Risk_Score']:
        if col in df.columns:
            target_variables.append(col)
    
    # Create the ML-ready dataframe
    ml_df = df[ml_features + target_variables]
    
    # Export to CSV
    ml_df.to_csv('safety_monitoring_ml_ready.csv', index=False)
    df.to_csv('safety_monitoring_cleaned.csv', index=False)
    
    print("\nML-ready dataset created and saved to 'safety_monitoring_ml_ready.csv'")
    print("Full cleaned dataset saved to 'safety_monitoring_cleaned.csv'")
    
    return ml_df

def main():
    """
    Main function to run the data cleaning and preparation pipeline
    """
    # File path to the dataset
    file_path = 'safety_monitoring.csv'
    
    # Clean the data
    cleaned_df = clean_safety_data(file_path)
    
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