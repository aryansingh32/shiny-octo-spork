import { toast } from "sonner";

// Base URL for the API
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:5000';

// Types
export interface User {
  id: string;
  name: string;
  age: number;
  address: string;
  contactPhone: string;
  emergencyContact: string;
  medicalConditions: string[];
  photo: string;
  preferences: {
    reminderNotifications: string;
    fontSize: string;
  };
}

export interface Reminder {
  id: string;
  userId: string;
  title: string;
  description: string;
  dateTime: string;
  completed: boolean;
  type: "medication" | "appointment" | "activity" | "other";
  recurrence: "once" | "daily" | "weekly" | "monthly";
  priority: "low" | "medium" | "high";
}

export interface Alert {
  id: string;
  userId: string;
  title: string;
  description: string;
  dateTime: string;
  severity: "info" | "warning" | "critical";
  acknowledged: boolean;
  source: "health" | "safety" | "system";
}

export interface HealthData {
  id: string;
  userId: string;
  type: "heartRate" | "bloodPressure" | "temperature" | "bloodSugar" | "oxygenLevel" | "weight" | "sleep";
  value: any;
  unit: string;
  dateTime: string;
  isAbnormal?: boolean;
}

// Generic fetch wrapper with error handling
async function fetchWithAuth<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log(`Making request to: ${url}`); // Log the URL being called

    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (response.status === 404) {
      throw new Error("The requested resource was not found.");
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error(`API Error Response:`, errorData); // Log the error response
      throw new Error(errorData.message || response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Error:`, error); // Log the error
    toast.error("An error occurred while communicating with the server");
    throw error;
  }
}

// Mock data generators
const mockUsers = Array(10)
  .fill(0)
  .map((_, i) => ({
    id: `user-${i + 1}`,
    name: [
      "Alice Johnson", 
      "Robert Smith", 
      "Maria Garcia", 
      "James Wilson", 
      "Patricia Brown", 
      "David Miller", 
      "Linda Davis",
      "William Moore",
      "Elizabeth Taylor",
      "John Anderson"
    ][i],
    age: 65 + Math.floor(Math.random() * 25),
    address: `${Math.floor(Math.random() * 1000) + 1} Main St, Anytown`,
    contactPhone: `(${Math.floor(Math.random() * 900) + 100}) ${Math.floor(Math.random() * 900) + 100}-${Math.floor(Math.random() * 9000) + 1000}`,
    emergencyContact: "Family Member",
    medicalConditions: [
      ["Hypertension", "Diabetes", "Arthritis"][Math.floor(Math.random() * 3)],
    ],
    photo: `/placeholder.svg`,
    preferences: {
      reminderNotifications: "both",
      fontSize: "large"
    }
  }));

const reminderTypes = ["medication", "appointment", "activity", "other"] as const;
const reminderPriorities = ["low", "medium", "high"] as const;
const reminderRecurrence = ["once", "daily", "weekly", "monthly"] as const;

const mockReminders = Array(20)
  .fill(0)
  .map((_, i) => ({
    id: `reminder-${i + 1}`,
    userId: mockUsers[Math.floor(Math.random() * mockUsers.length)].id,
    title: [
      "Take medication",
      "Doctor appointment",
      "Daily walk",
      "Blood pressure check",
      "Family visit"
    ][Math.floor(Math.random() * 5)],
    description: "Details about this reminder",
    dateTime: new Date(
      Date.now() + Math.floor(Math.random() * 7) * 86400000
    ).toISOString(),
    completed: Math.random() > 0.7,
    type: reminderTypes[Math.floor(Math.random() * reminderTypes.length)],
    recurrence: Math.random() > 0.5 
      ? reminderRecurrence[Math.floor(Math.random() * reminderRecurrence.length)]
      : "once",
    priority: reminderPriorities[Math.floor(Math.random() * reminderPriorities.length)]
  }));

const alertSeverities = ["info", "warning", "critical"] as const;
const alertSources = ["health", "safety", "system"] as const;

const mockAlerts = Array(8)
  .fill(0)
  .map((_, i) => ({
    id: `alert-${i + 1}`,
    userId: mockUsers[Math.floor(Math.random() * mockUsers.length)].id,
    title: [
      "Unusual heart rate detected",
      "Possible fall detected",
      "Medication reminder missed",
      "Room temperature too high",
      "Door left open",
      "No movement detected for 6 hours"
    ][Math.floor(Math.random() * 6)],
    description: "Details about this alert",
    dateTime: new Date(
      Date.now() - Math.floor(Math.random() * 3) * 86400000
    ).toISOString(),
    severity: alertSeverities[Math.floor(Math.random() * alertSeverities.length)],
    acknowledged: Math.random() > 0.3,
    source: alertSources[Math.floor(Math.random() * alertSources.length)]
  }));

const healthDataTypes = [
  "heartRate", 
  "bloodPressure", 
  "temperature", 
  "bloodSugar", 
  "oxygenLevel", 
  "weight", 
  "sleep"
] as const;

const mockHealthData: HealthData[] = [];

// Generate health data for each user for the past 7 days
mockUsers.forEach(user => {
  // For each health data type
  healthDataTypes.forEach(type => {
    // Generate data for the last 7 days
    for (let day = 0; day < 7; day++) {
      let value: any;
      let unit = "";
      let isAbnormal = false;
      
      // Generate realistic values based on type
      switch(type) {
        case "heartRate":
          value = 60 + Math.floor(Math.random() * 40);
          unit = "bpm";
          isAbnormal = value > 90 || value < 60;
          break;
        case "bloodPressure":
          value = {
            systolic: 110 + Math.floor(Math.random() * 40),
            diastolic: 70 + Math.floor(Math.random() * 20)
          };
          unit = "mmHg";
          isAbnormal = value.systolic > 140 || value.diastolic > 90;
          break;
        case "temperature":
          value = 36.1 + Math.random() * 2;
          unit = "°C";
          isAbnormal = value > 37.5 || value < 36.0;
          break;
        case "bloodSugar":
          value = 70 + Math.floor(Math.random() * 100);
          unit = "mg/dL";
          isAbnormal = value > 140 || value < 70;
          break;
        case "oxygenLevel":
          value = 92 + Math.floor(Math.random() * 9);
          unit = "%";
          isAbnormal = value < 95;
          break;
        case "weight":
          value = 60 + Math.floor(Math.random() * 40);
          unit = "kg";
          isAbnormal = false;
          break;
        case "sleep":
          value = 4 + Math.floor(Math.random() * 6);
          unit = "hours";
          isAbnormal = value < 6;
          break;
      }
      
      // Create the health data entry
      mockHealthData.push({
        id: `health-${user.id}-${type}-${day}`,
        userId: user.id,
        type,
        value,
        unit,
        dateTime: new Date(
          Date.now() - day * 86400000
        ).toISOString(),
        isAbnormal
      });
    }
  });
});

// API functions - these would make real API calls in production
// For now, they return mock data

// User API
export async function getUsers(): Promise<User[]> {
  return fetchWithAuth<User[]>("/api/users");
}

export async function getUser(id: string): Promise<User> {
  return fetchWithAuth<User>(`/api/users/${id}`);
}

export async function createUser(userData: Omit<User, "id">): Promise<User> {
  return fetchWithAuth<User>("/api/users", {
    method: "POST",
    body: JSON.stringify(userData),
  });
}

export async function updateUser(id: string, userData: Partial<User>): Promise<User> {
  return fetchWithAuth<User>(`/api/users/${id}`, {
    method: "PUT",
    body: JSON.stringify(userData),
  });
}

export async function deleteUser(id: string): Promise<boolean> {
  return fetchWithAuth<boolean>(`/api/users/${id}`, {
    method: "DELETE",
  });
}

// Reminder API
export async function getReminders(): Promise<Reminder[]> {
  return fetchWithAuth<Reminder[]>("/api/reminders");
}

export async function getUserReminders(userId: string): Promise<Reminder[]> {
  return fetchWithAuth<Reminder[]>(`/api/users/${userId}/reminders`);
}

export async function createReminder(reminderData: Omit<Reminder, "id">): Promise<Reminder> {
  return fetchWithAuth<Reminder>("/api/reminders", {
    method: "POST",
    body: JSON.stringify(reminderData),
  });
}

export async function updateReminder(id: string, reminderData: Partial<Reminder>): Promise<Reminder> {
  return fetchWithAuth<Reminder>(`/api/reminders/${id}`, {
    method: "PUT",
    body: JSON.stringify(reminderData),
  });
}

export async function deleteReminder(id: string): Promise<boolean> {
  return fetchWithAuth<boolean>(`/api/reminders/${id}`, {
    method: "DELETE",
  });
}

// Alert API
export async function getAlerts(): Promise<Alert[]> {
  return fetchWithAuth<Alert[]>("/api/alerts");
}

export async function getUserAlerts(userId: string): Promise<Alert[]> {
  return fetchWithAuth<Alert[]>(`/api/users/${userId}/alerts`);
}

export async function acknowledgeAlert(id: string): Promise<Alert> {
  return fetchWithAuth<Alert>(`/api/alerts/${id}/acknowledge`, {
    method: "POST",
  });
}

// Health data API
export async function getHealthData(userId: string, type?: string): Promise<HealthData[]> {
  const endpoint = type ? 
    `/api/users/${userId}/health-data/${type}` : 
    `/api/users/${userId}/health-data`;
  return fetchWithAuth<HealthData[]>(endpoint);
}

export async function getLatestHealthData(userId: string): Promise<Record<string, HealthData>> {
  return fetchWithAuth<Record<string, HealthData>>(`/api/users/${userId}/health-data/latest`);
}

export async function submitHealthData(userId: string, data: Omit<HealthData, "id" | "userId">): Promise<HealthData> {
  return fetchWithAuth<HealthData>(`/api/users/${userId}/health-data`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// System API
export async function checkHealth(): Promise<{ status: string }> {
  return fetchWithAuth<{ status: string }>("/api/health");
}

export async function getSystemInfo(): Promise<any> {
  return fetchWithAuth<any>("/api/info");
}
