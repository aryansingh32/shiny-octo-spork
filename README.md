# ElderlyCareUI

A comprehensive elderly care monitoring system with health and safety analytics, alerts, and reminders.

## Overview

ElderlyCareUI is an integrated elderly care management platform that combines real-time health monitoring, safety assessment, and medication reminders in a user-friendly interface. The system uses ML/AI agents to analyze health and safety data, detect anomalies, and provide timely alerts and recommendations to caregivers.

![ElderlyCareUI Dashboard](docs/dashboard_screenshot.png)

## Features

- **Health Monitoring**: Track vital signs like heart rate, blood pressure, temperature, oxygen saturation, and glucose levels with real-time analysis
- **Safety Monitoring**: Fall detection, location tracking, activity monitoring with anomaly detection
- **Smart Alerts**: Configurable alerts with different severity levels (emergency, urgent, routine)
- **Reminders System**: Medication, appointment, and activity reminders with acknowledgment tracking
- **User-friendly Dashboard**: Modern UI with clear visualization of health metrics and status
- **Multi-Agent Architecture**: Extensible system with specialized agents for different monitoring aspects

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- SQLite (included)

### Backend Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ElderlyCareUI.git
cd ElderlyCareUI
```

2. Create a virtual environment and install dependencies:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp backend.env.example backend.env
# Edit backend.env with your settings
```

4. Initialize the database:

```bash
python setup_db.py
```

5. Start the backend server:

```bash
python app.py
```

### Frontend Setup

1. Install dependencies:

```bash
cd ../frontend
npm install
```

2. Start the development server:

```bash
npm run dev
```

Visit `http://localhost:3000` to access the application.

## Usage

### Monitoring Dashboard

The main dashboard provides an overview of all monitored users with their current status. You can:

- View health and safety status at a glance with color-coded indicators
- See active alerts and pending reminders
- Access detailed individual reports

### Health Monitoring

The health monitoring section displays:

- Real-time vital sign measurements
- Historical trends with anomaly highlighting
- Medical thresholds with personalized ranges
- AI-generated insights and recommendations

### Safety Monitoring

The safety panel shows:

- Current location and movement status
- Fall detection alerts
- Activity level and inactivity durations
- Environmental risk assessments

### Reminders Management

The reminders section allows you to:

- Create, view and manage medication schedules
- Set up appointment reminders
- Configure daily activity prompts
- Track reminder acknowledgments

## Technical Architecture

### System Components

```
ElderlyCareUI/
├── backend/                # Python Flask backend
│   ├── app.py              # Main application entry point
│   ├── integration/        # Core system components
│   │   ├── main.py         # Main integration logic
│   │   ├── agent_coordinator.py  # Coordinates agents
│   │   ├── alert_system.py # Alert management
│   │   └── database.py     # Data access layer
│   ├── health_monitoring/  # Health monitoring subsystem
│   ├── safety_monitoring/  # Safety monitoring subsystem
│   └── daily_reminder/     # Reminder subsystem
├── src/                    # React/NextJS frontend
│   ├── app/                # Next.js pages
│   ├── components/         # UI components
│   │   ├── MLInsightsPanel.tsx  # ML insights display
│   │   └── AgentResultDisplay.tsx  # Agent results visualization
│   └── lib/                # Utility functions
└── docs/                   # Documentation
```

### Data Flow

1. **Data Collection**: Health and safety data is collected from sensors, wearables, or manual input
2. **Data Processing**: The `ElderlyCareSys` class processes incoming data via specialized methods
3. **Agent Analysis**: Data is passed to specialized agents via the `AgentCoordinator`
4. **Alert Generation**: Anomalies trigger alerts via the `AlertSystem`
5. **Frontend Display**: Results are pushed to the frontend and displayed in the `MLInsightsPanel` and `AgentResultDisplay` components

### Key Classes

- **ElderlyCareSys**: Main system class coordinating all components
- **Database**: Handles data persistence and retrieval
- **AgentCoordinator**: Manages specialized monitoring agents
- **AlertSystem**: Handles alert generation and notification

## Development Guide

### Adding a New Monitoring Agent

1. Create a new agent class in the appropriate directory
2. Implement the required interface methods (e.g., `analyze_health_data()`)
3. Register the agent in `agent_coordinator.py`
4. Update the frontend to display the new agent's results

### Health Data Processing Flow

The health data processing pipeline:

1. Raw data is received via API or sensors
2. `process_health_data()` method checks for anomalies using medical thresholds
3. Specialized health agents perform additional analysis
4. Results are merged and structured with status indicators
5. Alerts are generated for critical or moderate issues
6. Frontend displays color-coded metrics and recommendations

### Extending the System

The system is designed for extensibility:

- New metrics can be added to the health and safety data structures
- Additional agents can be integrated through the AgentCoordinator
- Custom thresholds can be configured per user
- New visualization components can be added to the frontend

## API Reference

### Health Monitoring Endpoints

- `GET /api/health/:userId` - Get latest health data for user
- `POST /api/health/:userId` - Submit new health data
- `GET /api/health/:userId/insights` - Get AI insights for health data

### Safety Monitoring Endpoints

- `GET /api/safety/:userId` - Get latest safety data
- `POST /api/safety/:userId` - Submit new safety data
- `GET /api/safety/:userId/insights` - Get AI insights for safety data

### Reminders Endpoints

- `GET /api/reminders/:userId` - Get reminders for user
- `POST /api/reminders/:userId` - Create new reminder
- `PUT /api/reminders/:userId/:reminderId` - Update reminder

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical thresholds based on standard geriatric care guidelines
- Safety monitoring protocols follow fall prevention best practices
- UI/UX designed for accessibility and ease of use for elderly care providers
