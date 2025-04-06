# Elderly Care System Backend Documentation

## Overview

The Elderly Care System is a comprehensive multi-agent AI-driven platform designed to monitor and support elderly individuals. The system implements a sophisticated architecture with specialized agents for health monitoring, safety monitoring, and daily reminders. These agents leverage machine learning models to provide intelligent and proactive care assistance.

## System Architecture

The backend consists of the following core components:

1. **Main Application (main.py)** - The central orchestrator that initializes and manages the entire system
2. **App Entry Point (app.py)** - Provides different interfaces (API, CLI, demo) to interact with the system
3. **Agent Coordinator** - Facilitates communication between specialized agents
4. **Specialized Agents**:
   - Health Monitoring Agent
   - Safety Monitoring Agent
   - Daily Reminder Agent
5. **Alert System** - Manages and distributes alerts based on severity
6. **Database** - Stores user data, health metrics, safety events, and reminders

## Core Components in Detail

### Main Application (main.py)

The `ElderlyCareSys` class in `main.py` serves as the central component of the system, responsible for:

- Initializing all system components (database, agents, alert system)
- Loading configuration from environment variables or config files
- Processing user data through the agent coordinator
- Managing user registration and system maintenance
- Generating user reports and system statistics
- Starting and stopping autonomous operations

### App Entry Point (app.py)

The `app.py` file provides multiple interfaces to interact with the system:

- **API Mode**: Runs a Flask server to expose RESTful endpoints for the frontend
- **CLI Mode**: Provides a command-line interface for administrators
- **Demo Mode**: Runs a demonstration of the system's capabilities

Each mode has proper error handling and logging to ensure robust operation.

### Agent Coordinator

The `AgentCoordinator` class is responsible for:

- Initializing all specialized agents
- Coordinating the data flow between agents
- Prioritizing outputs from different agents
- Managing inter-agent communication through message queues
- Scheduling periodic checks based on configured intervals
- Generating integrated recommendations based on all agents' outputs

### Database Structure

The system uses SQLite with the following main tables:

- **users**: Stores user information and preferences
- **health_data**: Records health metrics like vital signs
- **safety_data**: Stores safety-related events and fall detection
- **reminders**: Manages medication, appointment, and activity reminders
- **alerts**: Tracks emergency, urgent, and routine alerts
- **alert_recipients**: Maps alerts to their designated recipients
- **system_logs**: Records system events for auditing and debugging

## Machine Learning Implementation

### Data Preprocessing & Cleaning

Each agent has its own data preprocessing pipeline:

1. **Health Data Cleaning (`health_monetoring_clean.py`)**:
   - Normalizes vital signs (heart rate, blood pressure, etc.)
   - Handles missing values through imputation
   - Detects and removes outliers
   - Adds temporal features (hour, day, is_night)
   - Converts categorical data to numerical representations

2. **Safety Data Cleaning (`saftey_monetoring_clean.py`)**:
   - Normalizes movement and location data
   - Converts categorical variables (movement types, locations) to binary flags
   - Adds time-based features
   - Balances datasets for fall detection training

3. **Reminder Data Cleaning (`daily_remainder_cleaner.py`)**:
   - Processes temporal patterns in reminder compliance
   - Extracts reminder categories and priorities
   - Generates features for optimal timing prediction

### Agent Training & Models

#### Health Monitoring Agent

The `HealthMonitoringAgent` uses:

1. **Anomaly Detection**:
   - Trains an Isolation Forest model for unsupervised anomaly detection in vital signs
   - Features include normalized vital signs and temporal information
   
2. **Supervised Classification**:
   - Trains a Random Forest classifier to detect health anomalies
   - Uses labeled data with known health issues
   
3. **Severity Prediction**:
   - Uses a dedicated model to categorize health anomalies into severity levels (0-3)
   - Guides appropriate response based on severity

Key methods:
- `train_unsupervised_model()`: Trains the Isolation Forest
- `train_supervised_model()`: Trains the classifier
- `train_severity_model()`: Trains the severity predictor
- `detect_anomalies()`: Main detection pipeline
- `run_health_check()`: Performs complete health assessment

#### Safety Monitoring Agent

The `SafetyMonitoringAgent` implements:

1. **Fall Detection**:
   - Uses both unsupervised (Isolation Forest) and supervised (Random Forest) approaches
   - Analyzes movement patterns and sensor data to detect falls
   
2. **Risk Assessment**:
   - Predicts risk levels for potential safety issues
   - Considers location, movement patterns, and time of day
   
3. **Safety Pattern Analysis**:
   - Analyzes normal movement patterns to detect abnormal behavior

Key methods:
- `train_unsupervised_model()`: Trains anomaly detection
- `train_supervised_model()`: Trains fall detection
- `train_risk_level_model()`: Trains risk assessment
- `detect_falls()`: Main fall detection pipeline
- `monitor_safety()`: Complete safety monitoring workflow

#### Daily Reminder Agent

The `DailyReminderAgent` uses:

1. **Reminder Compliance Prediction**:
   - Predicts whether reminders will be acknowledged based on historical patterns
   - Uses reminder type, timing, and user behavior features
   
2. **Priority Optimization**:
   - Determines optimal priority levels for different reminders
   - Balances urgency with user responsiveness
   
3. **Optimal Timing Prediction**:
   - Predicts the most effective times to deliver reminders
   - Considers user daily routines and historical compliance

Key methods:
- `train_acknowledgment_model()`: Trains compliance prediction
- `train_priority_model()`: Trains priority optimizer
- `train_optimal_time_model()`: Trains timing predictor
- `predict_acknowledgment()`: Predicts reminder compliance
- `suggest_optimal_schedule()`: Recommends optimal reminder schedule

## Alert System

The `AlertSystem` class manages the creation, routing, and tracking of alerts:

1. **Alert Creation**:
   - Creates alerts from agent outputs
   - Assigns severity levels (emergency, urgent, routine, info)
   - Stores alerts in the database

2. **Alert Routing**:
   - Determines appropriate recipients based on severity
   - Supports multiple stakeholder types (caregivers, family, healthcare providers, emergency services)

3. **Notification Delivery**:
   - Sends notifications through appropriate channels
   - Supports voice alerts for immediate attention

4. **Alert Management**:
   - Tracks acknowledgment and resolution status
   - Provides history and statistics for reporting

Key methods:
- `create_alert()`: Creates a new alert
- `get_active_alerts()`: Retrieves unresolved alerts
- `acknowledge_alert()`: Marks an alert as acknowledged
- `resolve_alert()`: Marks an alert as resolved
- `send_voice_alert()`: Delivers an audible alert

## Environment Configuration

The `.env` file configures the system with the following settings:

- **Flask Configuration**: Environment, debug mode, port
- **Database Configuration**: Database path
- **Logging Configuration**: Log level and format
- **Agent Intervals**: How frequently each agent checks for issues
- **Alert Settings**: Cooldown periods for different alert types
- **Data Retention**: How long to keep historical data

## API Implementation

The REST API exposes the following main endpoints:

1. **User Management**:
   - `/api/users` - CRUD operations for users
   - `/api/users/<id>/report` - Generate comprehensive user reports

2. **Health Monitoring**:
   - `/api/health/data` - Submit health data
   - `/api/health/anomalies` - Get detected anomalies
   - `/api/health/report/<user_id>` - Get health reports

3. **Safety Monitoring**:
   - `/api/safety/data` - Submit safety data
   - `/api/safety/falls` - Get detected falls
   - `/api/safety/risks` - Get risk assessments

4. **Reminders**:
   - `/api/reminders` - CRUD operations for reminders
   - `/api/reminders/upcoming` - Get upcoming reminders
   - `/api/reminders/optimize` - Optimize reminder schedule

5. **Alerts**:
   - `/api/alerts` - Get and create alerts
   - `/api/alerts/acknowledge` - Acknowledge alerts
   - `/api/alerts/resolve` - Resolve alerts

## Voice Reminder Service

The system includes a voice reminder service that:

1. Converts text reminders to speech using pyttsx3
2. Delivers audible reminders at scheduled times
3. Supports different voice profiles and speech rates
4. Provides acknowledgment mechanisms

## Communication Between Components

The system implements a message-based communication architecture:

1. **AgentMessage Class**: Standardized message format for inter-agent communication
2. **Message Queues**: Priority queues for message passing
3. **Coordinator Processing**: Central processing of all inter-agent messages
4. **Alert Generation**: Automatic alert creation based on agent findings

## Autonomous Operation

The system supports continuous monitoring through:

1. **Autonomous Threads**: Background threads for each agent
2. **Periodic Checks**: Configured intervals for health, safety, and reminder checks
3. **Dynamic Prioritization**: Adjusts check frequency based on detected issues
4. **Persistent Database**: Stores all findings for later analysis

## Error Handling and Logging

The system implements robust error handling:

1. **Exception Catching**: All critical operations are wrapped in try-except blocks
2. **Rotating Logs**: Log files are rotated to prevent excessive size
3. **Error Levels**: Different severity levels for logging (INFO, WARNING, ERROR)
4. **Traceback Capture**: Full error context is captured for debugging

## Conclusion

The Elderly Care System backend is a sophisticated multi-agent system that integrates health monitoring, safety monitoring, and daily reminders with machine learning capabilities. The architecture ensures scalability, robustness, and intelligent care provision for elderly individuals. 