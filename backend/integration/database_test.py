import sqlite3

def initialize_database(db_path="elderly_care.db"):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Drop health_data table if it exists
    cursor.execute("DROP TABLE IF EXISTS health_data")

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        severity TEXT NOT NULL,
        source_agent TEXT NOT NULL,
        resolved BOOLEAN DEFAULT 0
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        reminder_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        scheduled_time TIMESTAMP NOT NULL,
        priority TEXT NOT NULL,
        type TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS health_data (
        user_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        heart_rate INTEGER NOT NULL,
        blood_pressure TEXT NOT NULL,
        temperature REAL NOT NULL,
        is_abnormal BOOLEAN NOT NULL
    )
    """)

    # Insert sample data
    cursor.executemany("""
    INSERT OR IGNORE INTO users (user_id, name, age) VALUES (?, ?, ?)
    """, [
        (1, 'Alice Johnson', 86),
        (2, 'Robert Smith', 81),
        (3, 'Maria Garcia', 71)
    ])
    cursor.executemany("""
    INSERT OR IGNORE INTO alerts (alert_id, user_id, message, severity, source_agent, resolved) VALUES (?, ?, ?, ?, ?, ?)
    """, [
        ('alert1', 1, 'High blood pressure detected', 'urgent', 'health_monitoring', 0),
        ('alert2', 2, 'Fall detected', 'emergency', 'safety_monitoring', 0),
        ('alert3', 3, 'No movement detected for 6 hours', 'warning', 'safety_monitoring', 0)
    ])
    cursor.executemany("""
    INSERT OR IGNORE INTO reminders (reminder_id, user_id, message, scheduled_time, priority, type) VALUES (?, ?, ?, ?, ?, ?)
    """, [
        ('reminder1', 1, 'Doctor appointment', '2025-04-06 10:00:00', 'high', 'appointment'),
        ('reminder2', 2, 'Daily walk', '2025-04-05 08:00:00', 'low', 'activity'),
        ('reminder3', 3, 'Blood pressure check', '2025-04-06 18:00:00', 'high', 'activity')
    ])

    connection.commit()
    connection.close()

if __name__ == "__main__":
    initialize_database()
    print("Database initialized with sample data.")