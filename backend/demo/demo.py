import os
import sys
import time
import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from integration.main import ElderlyCareSys
from integration.database import Database
from integration.agent_coordinator import AgentCoordinator
from integration.alert_system import AlertSystem

# Create demo directory if it doesn't exist
demo_dir = Path(__file__).parent
demo_data_dir = demo_dir / "data"
demo_data_dir.mkdir(exist_ok=True)

class DemoData:
    """Class for generating test data for the Elderly Care System"""
    
    def __init__(self):
        """Initialize the data generator"""
        self.current_date = datetime.datetime.now()
        
    def generate_user(self):
        """Generate a user profile"""
        names = ["Alice Johnson", "Robert Smith", "Mary Williams", "James Brown", "Patricia Davis"]
        ages = list(range(65, 95))
        
        return {
            "name": random.choice(names),
            "age": random.choice(ages),
            "preferences": {
                "medication_reminder_frequency": random.choice(["high", "medium", "low"]),
                "activity_level": random.choice(["active", "moderate", "sedentary"]),
                "sleep_hours": random.randint(6, 9),
                "preferred_reminder_time": f"{random.randint(7, 10)}:00"
            }
        }
    
    def generate_health_data(self, days=7, readings_per_day=4):
        """Generate health monitoring data over a period of days"""
        data = []
        base_date = self.current_date - datetime.timedelta(days=days)
        
        # Base readings with some normal variations
        base_heart_rate = 72
        base_blood_pressure_sys = 130
        base_blood_pressure_dia = 85
        base_temperature = 36.6
        base_oxygen = 96
        base_glucose = 110
        
        # Generate readings for each day
        for day in range(days):
            for reading in range(readings_per_day):
                timestamp = base_date + datetime.timedelta(days=day, hours=6*reading)
                
                # Add some daily pattern and random noise
                hour_factor = 1 + (0.1 * np.sin(timestamp.hour / 12 * np.pi))
                
                # Occasionally insert an anomaly
                anomaly = random.random() < 0.05
                
                if anomaly:
                    heart_rate = base_heart_rate * hour_factor * random.uniform(1.3, 1.6)
                    blood_pressure_sys = base_blood_pressure_sys * hour_factor * random.uniform(1.2, 1.4)
                    blood_pressure_dia = base_blood_pressure_dia * hour_factor * random.uniform(1.2, 1.4)
                    temperature = base_temperature * random.uniform(1.02, 1.04)
                    oxygen = base_oxygen * random.uniform(0.85, 0.9)
                    glucose = base_glucose * random.uniform(1.5, 2.0)
                else:
                    heart_rate = base_heart_rate * hour_factor * random.uniform(0.9, 1.1)
                    blood_pressure_sys = base_blood_pressure_sys * hour_factor * random.uniform(0.95, 1.05)
                    blood_pressure_dia = base_blood_pressure_dia * hour_factor * random.uniform(0.95, 1.05)
                    temperature = base_temperature * random.uniform(0.99, 1.01)
                    oxygen = base_oxygen * random.uniform(0.97, 1.0)
                    glucose = base_glucose * random.uniform(0.9, 1.1)
                
                data.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "heart_rate": round(heart_rate, 1),
                    "blood_pressure_systolic": round(blood_pressure_sys, 1),
                    "blood_pressure_diastolic": round(blood_pressure_dia, 1),
                    "temperature": round(temperature, 1),
                    "oxygen_saturation": round(oxygen, 1),
                    "glucose_level": round(glucose, 1)
                })
                
        return data
    
    def generate_safety_data(self, days=7, readings_per_day=24):
        """Generate safety monitoring data over a period of days"""
        data = []
        base_date = self.current_date - datetime.timedelta(days=days)
        
        # Home locations
        locations = ["bedroom", "bathroom", "kitchen", "living_room", "hallway"]
        
        # Activity patterns for a typical elderly person
        activity_patterns = {
            0: "sleeping", 1: "sleeping", 2: "sleeping", 3: "sleeping", 4: "sleeping", 
            5: "sleeping", 6: "bathroom", 7: "kitchen", 8: "kitchen", 9: "living_room",
            10: "living_room", 11: "kitchen", 12: "kitchen", 13: "living_room", 14: "living_room",
            15: "living_room", 16: "kitchen", 17: "kitchen", 18: "living_room", 19: "living_room",
            20: "living_room", 21: "bathroom", 22: "bedroom", 23: "sleeping"
        }
        
        # Generate readings for each day
        for day in range(days):
            for hour in range(readings_per_day):
                timestamp = base_date + datetime.timedelta(days=day, hours=hour)
                
                # Determine expected location and activity based on hour
                expected_activity = activity_patterns.get(hour, "unknown")
                
                # For sleeping hours, set bedroom as location
                if expected_activity == "sleeping":
                    location = "bedroom"
                else:
                    location = expected_activity if expected_activity in locations else random.choice(locations)
                
                # Random movement metrics
                movement_speed = 0 if expected_activity == "sleeping" else random.uniform(0.5, 2.0)
                
                # Fall detection (rare random event)
                fall_detected = random.random() < 0.01
                unusual_behavior = random.random() < 0.05
                
                # If it's middle of the night and not in bedroom, mark as unusual
                if 0 <= hour <= 4 and location != "bedroom" and not expected_activity == "sleeping":
                    unusual_behavior = True
                
                # If fall detected, add details
                fall_details = None
                if fall_detected:
                    fall_details = {
                        "impact_force": random.uniform(5.0, 15.0),
                        "duration_on_ground": random.randint(10, 300)
                    }
                
                data.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "location": location,
                    "movement_detected": movement_speed > 0.1,
                    "movement_speed": round(movement_speed, 2),
                    "fall_detected": fall_detected,
                    "unusual_behavior": unusual_behavior,
                    "fall_details": fall_details,
                    "expected_activity": expected_activity
                })
                
        return data
    
    def generate_reminders(self, days=7):
        """Generate reminder data over a period of days"""
        data = []
        base_date = self.current_date - datetime.timedelta(days=days)
        
        # Common medications and schedules
        medications = [
            {"name": "Lisinopril", "schedule": ["08:00"], "type": "medication"},
            {"name": "Metformin", "schedule": ["08:00", "18:00"], "type": "medication"},
            {"name": "Atorvastatin", "schedule": ["20:00"], "type": "medication"},
            {"name": "Levothyroxine", "schedule": ["07:00"], "type": "medication"}
        ]
        
        # Appointments and activities
        appointments = [
            {"name": "Doctor appointment", "type": "appointment"},
            {"name": "Physical therapy", "type": "appointment"},
            {"name": "Blood test", "type": "appointment"}
        ]
        
        activities = [
            {"name": "Morning walk", "schedule": ["09:00"], "type": "activity"},
            {"name": "Lunch", "schedule": ["12:00"], "type": "activity"},
            {"name": "Dinner", "schedule": ["18:00"], "type": "activity"},
            {"name": "Social call", "type": "activity"}
        ]
        
        # Add daily medications
        for day in range(days):
            current_date = base_date + datetime.timedelta(days=day)
            
            # Add medications
            for med in medications:
                for schedule_time in med["schedule"]:
                    hour, minute = map(int, schedule_time.split(":"))
                    reminder_time = current_date.replace(hour=hour, minute=minute)
                    
                    # Add some randomness to acknowledgment
                    acknowledged = random.random() < 0.9  # 90% compliance rate
                    acknowledgment_time = None
                    
                    if acknowledged:
                        # Acknowledge within 0-30 minutes
                        ack_delay = datetime.timedelta(minutes=random.randint(0, 30))
                        acknowledgment_time = (reminder_time + ack_delay).strftime("%Y-%m-%d %H:%M:%S")
                    
                    data.append({
                        "title": f"Take {med['name']}",
                        "time": reminder_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "type": med["type"],
                        "priority": "high",
                        "acknowledged": acknowledged,
                        "acknowledgment_time": acknowledgment_time
                    })
            
            # Add activities for this day
            for activity in activities:
                if "schedule" in activity:
                    for schedule_time in activity["schedule"]:
                        hour, minute = map(int, schedule_time.split(":"))
                        reminder_time = current_date.replace(hour=hour, minute=minute)
                        
                        acknowledged = random.random() < 0.8  # 80% compliance
                        acknowledgment_time = None
                        
                        if acknowledged:
                            ack_delay = datetime.timedelta(minutes=random.randint(0, 45))
                            acknowledgment_time = (reminder_time + ack_delay).strftime("%Y-%m-%d %H:%M:%S")
                        
                        data.append({
                            "title": activity["name"],
                            "time": reminder_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "type": activity["type"],
                            "priority": "medium",
                            "acknowledged": acknowledged,
                            "acknowledgment_time": acknowledgment_time
                        })
            
            # Add random appointments (less frequent)
            if random.random() < 0.2:  # 20% chance of having an appointment on any day
                appointment = random.choice(appointments)
                hour = random.randint(9, 16)
                reminder_time = current_date.replace(hour=hour, minute=0)
                
                acknowledged = random.random() < 0.95  # 95% compliance for appointments
                acknowledgment_time = None
                
                if acknowledged:
                    ack_delay = datetime.timedelta(minutes=random.randint(0, 20))
                    acknowledgment_time = (reminder_time + ack_delay).strftime("%Y-%m-%d %H:%M:%S")
                
                data.append({
                    "title": appointment["name"],
                    "time": reminder_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": appointment["type"],
                    "priority": "high",
                    "acknowledged": acknowledged,
                    "acknowledgment_time": acknowledgment_time
                })
                
        return data
    
    def save_demo_data(self):
        """Generate and save all demo data to files"""
        # Generate user data
        user = self.generate_user()
        
        # Generate datasets
        health_data = self.generate_health_data()
        safety_data = self.generate_safety_data()
        reminder_data = self.generate_reminders()
        
        # Save to files
        with open(demo_data_dir / "user.json", "w") as f:
            import json
            json.dump(user, f, indent=2)
            
        pd.DataFrame(health_data).to_csv(demo_data_dir / "health_data.csv", index=False)
        pd.DataFrame(safety_data).to_csv(demo_data_dir / "safety_data.csv", index=False)
        pd.DataFrame(reminder_data).to_csv(demo_data_dir / "reminder_data.csv", index=False)
        
        return {
            "user": user,
            "health_data": health_data,
            "safety_data": safety_data,
            "reminder_data": reminder_data
        }

def run_system_test(system, user_id, data):
    """Run a comprehensive test of the system with the provided data"""
    logger.info("Starting comprehensive system test")
    
    # 1. Process health data
    logger.info("Processing health data...")
    for entry in data["health_data"]:
        system.database.store_health_data(user_id, entry)
    
    health_results = system.process_health_data(user_id)
    logger.info(f"Health processing results: {health_results}")
    
    # 2. Process safety data
    logger.info("Processing safety data...")
    for entry in data["safety_data"]:
        system.database.store_safety_data(user_id, entry)
    
    safety_results = system.process_safety_data(user_id)
    logger.info(f"Safety processing results: {safety_results}")
    
    # 3. Process reminder data
    logger.info("Processing reminder data...")
    for entry in data["reminder_data"]:
        system.database.add_reminder(user_id, entry)
    
    reminder_results = system.process_reminder_data(user_id)
    logger.info(f"Reminder processing results: {reminder_results}")
    
    # 4. Process everything together
    logger.info("Processing all data together...")
    integrated_results = system.process_data_for_user(user_id)
    logger.info(f"Integrated processing results: {integrated_results}")
    
    # 5. Check alerts generated
    alerts = system.alert_system.get_active_alerts(user_id)
    logger.info(f"Generated alerts: {alerts}")
    
    # 6. Generate dashboard data
    dashboard_data = system.get_dashboard_data(user_id)
    logger.info(f"Dashboard data generated")
    
    return {
        "health_results": health_results,
        "safety_results": safety_results,
        "reminder_results": reminder_results,
        "integrated_results": integrated_results,
        "alerts": alerts,
        "dashboard_data": dashboard_data
    }

def visualize_results(results, output_dir=None):
    """Create visualizations of the test results"""
    if output_dir is None:
        output_dir = demo_data_dir / "results"
        output_dir.mkdir(exist_ok=True)
    
    # 1. Visualize health anomalies
    if "anomalies" in results["health_results"]:
        anomalies = results["health_results"]["anomalies"]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(anomalies)), [a.get("severity", 0) for a in anomalies])
        plt.xlabel("Anomaly Index")
        plt.ylabel("Severity Score")
        plt.title("Health Anomalies by Severity")
        plt.savefig(output_dir / "health_anomalies.png")
        
    # 2. Visualize safety risks
    if "risks" in results["safety_results"]:
        risks = results["safety_results"]["risks"]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(risks)), [r.get("risk_level", 0) for r in risks])
        plt.xlabel("Risk Index")
        plt.ylabel("Risk Level")
        plt.title("Safety Risks by Level")
        plt.savefig(output_dir / "safety_risks.png")
        
    # 3. Visualize reminder compliance
    if "compliance" in results["reminder_results"]:
        compliance_rate = results["reminder_results"]["compliance"]
        categories = ["Medication", "Appointment", "Activity"]
        rates = [
            compliance_rate.get("medication", 0),
            compliance_rate.get("appointment", 0),
            compliance_rate.get("activity", 0)
        ]
        
        plt.figure(figsize=(8, 6))
        plt.bar(categories, rates)
        plt.xlabel("Reminder Type")
        plt.ylabel("Compliance Rate (%)")
        plt.title("Reminder Compliance by Type")
        plt.ylim(0, 100)
        plt.savefig(output_dir / "reminder_compliance.png")
        
    # 4. Visualize alert distribution
    alert_types = [alert.get("severity", "unknown") for alert in results["alerts"]]
    alert_counts = {
        "emergency": alert_types.count("emergency"),
        "urgent": alert_types.count("urgent"),
        "routine": alert_types.count("routine")
    }
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        alert_counts.values(),
        labels=alert_counts.keys(),
        autopct='%1.1f%%',
        colors=['red', 'orange', 'blue']
    )
    plt.title("Alert Distribution by Severity")
    plt.savefig(output_dir / "alert_distribution.png")
    
    logger.info(f"Visualizations saved to {output_dir}")

def run_demo():
    """Main demonstration function"""
    logger.info("Starting Elderly Care System Demonstration")
    
    # Initialize the system
    system = ElderlyCareSys()
    
    # Step 1: Generate demo data
    logger.info("Generating demonstration data...")
    data_generator = DemoData()
    data = data_generator.save_demo_data()
    
    # Step 2: Add a demo user
    user_data = data["user"]
    logger.info(f"Adding demo user: {user_data['name']}, {user_data['age']} years old")
    user_id = system.database.add_user(user_data["name"], user_data["age"], user_data["preferences"])
    logger.info(f"User added with ID: {user_id}")
    
    # Step 3: Add stakeholders
    logger.info("Adding stakeholders...")
    # Add a family member
    family_id = system.database.add_stakeholder({
        "name": "John Smith",
        "role": "family",
        "relationship": "son",
        "contact": {
            "phone": "555-123-4567",
            "email": "john.smith@example.com"
        }
    })
    
    # Add a healthcare provider
    doctor_id = system.database.add_stakeholder({
        "name": "Dr. Emily Johnson",
        "role": "healthcare",
        "specialty": "geriatrics",
        "contact": {
            "phone": "555-987-6543",
            "email": "dr.johnson@example.com"
        }
    })
    
    # Link stakeholders to the user
    system.database.link_stakeholder_to_user(user_id, family_id, {
        "emergency": True,
        "urgent": True,
        "routine": True
    })
    
    system.database.link_stakeholder_to_user(user_id, doctor_id, {
        "emergency": True,
        "urgent": True,
        "routine": False
    })
    
    # Step 4: Run the system test
    logger.info("Running comprehensive system test...")
    results = run_system_test(system, user_id, data)
    
    # Step 5: Visualize the results
    logger.info("Generating result visualizations...")
    visualize_results(results)
    
    # Step 6: Show demonstration summary
    print("\n" + "="*50)
    print("ELDERLY CARE SYSTEM DEMONSTRATION SUMMARY")
    print("="*50)
    print(f"User: {user_data['name']}, {user_data['age']} years old")
    print(f"Data processed: {len(data['health_data'])} health entries, {len(data['safety_data'])} safety entries, {len(data['reminder_data'])} reminders")
    print(f"Alerts generated: {len(results['alerts'])}")
    print(f"Health anomalies detected: {len(results['health_results'].get('anomalies', []))}")
    print(f"Safety risks identified: {len(results['safety_results'].get('risks', []))}")
    print(f"Reminder compliance rate: {results['reminder_results'].get('overall_compliance', 0):.1f}%")
    print("\nResults and visualizations saved to:", demo_data_dir)
    print("="*50)
    
    logger.info("Demonstration completed successfully")
    return results

if __name__ == "__main__":
    run_demo()