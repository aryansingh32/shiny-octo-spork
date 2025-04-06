import { toast } from "sonner";
import { User, Alert, HealthData } from "./api";

// Base URL for the backend API
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:5000';

// Interface for safety data
export interface SafetyData {
  id: string;
  userId: string;
  type: string;
  timestamp: string;
  data: Record<string, any>;
}

// Interface for safety alerts
export interface SafetyAlert {
  id: string;
  userId: string;
  type: string;
  severity: "low" | "medium" | "high";
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

// Interface for health anomalies
export interface HealthAnomaly {
  id: string;
  userId: string;
  type: string;
  metric: string;
  value: any;
  expectedRange: {
    min: number;
    max: number;
  };
  timestamp: string;
  severity: "low" | "medium" | "high";
  acknowledged: boolean;
}

// Interface for health overview
export interface HealthOverview {
  userId: string;
  status: "normal" | "caution" | "alert";
  metrics: {
    [key: string]: {
      current: number;
      trend: "stable" | "improving" | "declining";
      unit: string;
    };
  };
  lastUpdated: string;
}

// Interface for health report
export interface HealthReport {
  userId: string;
  startDate: string;
  endDate: string;
  metrics: {
    [key: string]: {
      values: Array<{
        value: number;
        timestamp: string;
      }>;
      average: number;
      min: number;
      max: number;
      trend: "stable" | "improving" | "declining";
    };
  };
  anomalies: HealthAnomaly[];
}

// Interface for stakeholder
export interface Stakeholder {
  id: string;
  name: string;
  role: "family" | "caregiver" | "healthcare" | "other";
  contactInfo: {
    email?: string;
    phone?: string;
  };
  alertPreferences?: {
    health: boolean;
    safety: boolean;
    reminders: boolean;
    notificationMethod: "email" | "sms" | "both" | "none";
  };
}

// Interface for dashboard data
export interface DashboardData {
  userId: string;
  healthSummary: {
    status: "normal" | "caution" | "alert";
    metrics: Record<string, {
      current: number;
      trend: "stable" | "improving" | "declining";
      unit: string;
    }>;
  };
  safetySummary: {
    status: "normal" | "caution" | "alert";
    recentIncidents: number;
  };
  remindersSummary: {
    upcoming: number;
    missed: number;
    completed: number;
  };
  alertsSummary: {
    active: number;
    bySeverity: {
      low: number;
      medium: number;
      high: number;
    };
    byType: Record<string, number>;
  };
}

// For the backend API, we need a different Reminder interface
export interface Reminder {
  id: number;
  user_id: number;
  user_name?: string;
  title: string;
  description: string;
  reminder_type?: string;
  type?: string;
  scheduled_time?: string;
  dateTime?: string;
  status: string;
  priority: string;
  recurrence: string;
  created_at: string;
  is_acknowledged?: number;
  acknowledged_time?: string | null;
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
        "Accept": "application/json",
        ...options?.headers,
      },
      credentials: 'include', // Include credentials for CORS
      mode: 'cors', // Explicitly set CORS mode
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

// System Management API
export async function checkBackendHealth(): Promise<{ status: string }> {
  return fetchWithAuth<{ status: string }>("/api/health");
}

export async function getApiInfo(): Promise<any> {
  return fetchWithAuth<any>("/");
}

// Health Monitoring API
export async function getHealthAnomalies(): Promise<HealthAnomaly[]> {
  return fetchWithAuth<HealthAnomaly[]>("/api/health/anomalies");
}

export async function getHealthOverview(): Promise<HealthOverview> {
  return fetchWithAuth<HealthOverview>("/api/health/overview");
}

export async function getHealthReport(userId: string): Promise<HealthReport> {
  return fetchWithAuth<HealthReport>(`/api/health/report/${userId}`);
}

export async function getHealthData(userId: string, metric: string): Promise<any[]> {
  return fetchWithAuth<any[]>(`/api/users/${userId}/health-data/${metric}`);
}

export async function getLatestHealthData(userId: string): Promise<Record<string, any>> {
  return fetchWithAuth<Record<string, any>>(`/api/users/${userId}/health-data/latest`);
}

export async function submitHealthData(userId: string, data: any): Promise<{ success: boolean, results?: any }> {
  try {
    const response = await fetchWithAuth<{ success: boolean, message: string, results?: any }>(`/api/users/${userId}/health-data`, {
    method: "POST",
    body: JSON.stringify(data),
  });
    
    console.log("Health data submission response:", response);
    return {
      success: response.success,
      results: response.results || null
    };
  } catch (error) {
    console.error("Error submitting health data:", error);
    return { success: false };
  }
}

// Safety Monitoring API
export async function getSafetyAlerts(): Promise<SafetyAlert[]> {
  return fetchWithAuth<SafetyAlert[]>("/api/safety/alerts");
}

export async function submitSafetyData(userId: string, data: any): Promise<{ success: boolean, results?: any }> {
  try {
    const response = await fetchWithAuth<{ success: boolean, message: string, results?: any }>(`/api/users/${userId}/safety-data`, {
    method: "POST",
    body: JSON.stringify(data),
  });
    
    console.log("Safety data submission response:", response);
    return {
      success: response.success,
      results: response.results || null
    };
  } catch (error) {
    console.error("Error submitting safety data:", error);
    return { success: false };
  }
}

// Daily Reminder API
export async function getRemindersFromBackend(): Promise<Reminder[]> {
  console.log("Fetching all reminders...");
  const response = await fetchWithAuth<{ success: boolean; reminders: Reminder[] }>("/api/reminders");
  console.log("Reminders response:", response);
  return response.reminders || [];
}

export async function getUpcomingReminders(): Promise<Reminder[]> {
  return fetchWithAuth<Reminder[]>("/api/reminders/upcoming");
}

export async function getRemindersCount(): Promise<{ 
  total: number; 
  upcoming: number; 
  missed: number; 
  completed: number 
}> {
  console.log("Fetching reminders count...");
  const response = await fetchWithAuth<{ 
    success: boolean;
    total: number; 
    upcoming: number; 
    missed: number; 
    completed: number 
  }>("/api/reminders/count");
  
  console.log("Reminders count response:", response);
  
  return {
    total: response.total || 0,
    upcoming: response.upcoming || 0,
    missed: response.missed || 0,
    completed: response.completed || 0
  };
}

export async function getUserRemindersFromBackend(userId: string): Promise<Reminder[]> {
  if (!userId) {
    console.log("No userId provided, returning empty array");
    return [];
  }
  console.log(`Fetching reminders for user ${userId}...`);
  const response = await fetchWithAuth<{ success: boolean; reminders: Reminder[] }>(`/api/users/${userId}/reminders`);
  console.log(`Reminders response for user ${userId}:`, response);
  return response.reminders || [];
}

export async function addReminder(userId: string, reminderData: Omit<Reminder, "id">): Promise<Reminder> {
  return fetchWithAuth<Reminder>(`/api/users/${userId}/reminders`, {
    method: "POST",
    body: JSON.stringify(reminderData),
  });
}

export async function updateReminderInBackend(
  userId: string,
  reminderId: string,
  reminderData: Partial<Reminder>
): Promise<Reminder> {
  console.log(`Updating reminder ${reminderId} for user ${userId} with data:`, JSON.stringify(reminderData));
  try {
    const result = await fetchWithAuth<{ success: boolean; reminder: Reminder }>(
      `/api/users/${userId}/reminders/${reminderId}`,
      {
    method: "PUT",
    body: JSON.stringify(reminderData),
      }
    );
    console.log(`Update result:`, result);
    return result.reminder;
  } catch (error) {
    console.error(`Error updating reminder:`, error);
    throw error;
  }
}

export async function acknowledgeReminder(
  userId: string,
  reminderId: string
): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>(
    `/api/users/${userId}/reminders/${reminderId}/acknowledge`,
    {
      method: "POST",
    }
  );
}

// Alert Management API
export async function getActiveAlerts(): Promise<Alert[]> {
  try {
    const response = await fetchWithAuth<{ success: boolean; alerts: any[] }>("/api/alerts");
    console.log("Raw alerts response:", response);
    
    if (!response.alerts) return [];
    
    // Map backend data to our frontend Alert interface format
    return response.alerts.map(alert => ({
      id: String(alert.id || alert.alert_id),
      userId: String(alert.user_id),
      title: alert.type || alert.source_agent || 'Alert',
      description: alert.message || '',
      dateTime: alert.timestamp || alert.created_at || new Date().toISOString(),
      severity: mapBackendSeverity(alert.severity),
      acknowledged: Boolean(alert.is_acknowledged || alert.acknowledged),
      source: mapSourceAgent(alert.source_agent || 'system')
    }));
  } catch (error) {
    console.error("Error fetching alerts:", error);
    return [];
  }
}

// Helper function to map backend severity levels to frontend severity levels
function mapBackendSeverity(severity: string): "info" | "warning" | "critical" {
  switch (severity?.toLowerCase()) {
    case 'emergency':
    case 'critical':
    case 'high':
      return 'critical';
    case 'urgent':
    case 'warning':
    case 'medium':
      return 'warning';
    case 'routine':
    case 'info':
    case 'low':
      return 'info';
    default:
      return 'warning';
  }
}

// Helper function to map backend source agent to frontend source
function mapSourceAgent(sourceAgent: string): "health" | "safety" | "system" {
  switch (sourceAgent?.toLowerCase()) {
    case 'health_monitoring':
    case 'health':
      return 'health';
    case 'safety_monitoring':
    case 'safety':
      return 'safety';
    default:
      return 'system';
  }
}

export async function getAlertsRequiringAttention(): Promise<Alert[]> {
  return fetchWithAuth<Alert[]>("/api/alerts/attention");
}

export async function getAlertsCount(): Promise<{ 
  total: number; 
  byType: Record<string, number>;
  bySeverity: { low: number; medium: number; high: number }
}> {
  return fetchWithAuth<{ 
    total: number; 
    byType: Record<string, number>;
    bySeverity: { low: number; medium: number; high: number }
  }>("/api/alerts/count");
}

export async function acknowledgeAlertInBackend(alertId: string): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>(`/api/alerts/${alertId}/acknowledge`, {
    method: "POST",
  });
}

export async function resolveAlert(alertId: string): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>(`/api/alerts/${alertId}/resolve`, {
    method: "POST",
  });
}

// User Management API
export async function getUsersFromBackend(): Promise<User[]> {
  const response = await fetchWithAuth<{ success: boolean; users: any[] }>("/api/users");
  
  // Transform backend fields to frontend format
  return (response.users || []).map(user => ({
    id: user.id.toString(),
    name: user.name,
    age: user.age,
    address: user.address || '',
    contactPhone: user.contact_phone || user.contactPhone || '',
    emergencyContact: user.emergency_contact || user.emergencyContact || '',
    medicalConditions: user.medical_conditions ? 
      (typeof user.medical_conditions === 'string' ? 
        JSON.parse(user.medical_conditions) : user.medical_conditions) : [],
    photo: user.photo || '/placeholder.svg',
    preferences: user.preferences ? 
      (typeof user.preferences === 'string' ? 
        JSON.parse(user.preferences) : user.preferences) : {
        reminderNotifications: 'both',
        fontSize: 'large'
      }
  }));
}

export async function getUsersCount(): Promise<{ total: number }> {
  const response = await fetchWithAuth<{ success: boolean; count: number }>("/api/users/count");
  return { total: response.count || 0 };
}

export async function getUserFromBackend(userId: string): Promise<User> {
  const response = await fetchWithAuth<{ success: boolean; user: User }>(`/api/users/${userId}`);
  return response.user;
}

export async function createUserInBackend(userData: Omit<User, "id">): Promise<User> {
  return fetchWithAuth<User>("/api/users", {
    method: "POST",
    body: JSON.stringify(userData),
  });
}

export async function updateUserInBackend(userId: string, userData: Partial<User>): Promise<User> {
  return fetchWithAuth<User>(`/api/users/${userId}`, {
    method: "PUT",
    body: JSON.stringify(userData),
  });
}

export async function deleteUserFromBackend(userId: string): Promise<boolean> {
  return fetchWithAuth<boolean>(`/api/users/${userId}`, {
    method: "DELETE",
  });
}

// Stakeholder Management API
export async function addStakeholder(stakeholderData: Omit<Stakeholder, "id">): Promise<Stakeholder> {
  return fetchWithAuth<Stakeholder>("/api/stakeholders", {
    method: "POST",
    body: JSON.stringify(stakeholderData),
  });
}

export async function linkStakeholderToUser(
  userId: string,
  stakeholderId: string,
  preferences?: {
    health: boolean;
    safety: boolean;
    reminders: boolean;
    notificationMethod: "email" | "sms" | "both" | "none";
  }
): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>(`/api/users/${userId}/stakeholders`, {
    method: "POST",
    body: JSON.stringify({ stakeholderId, preferences }),
  });
}

// Dashboard and Insights API
export async function processAllData(userId: string): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>(`/api/process-all/${userId}`, {
    method: "POST",
  });
}

export async function getDashboardData(userId: string): Promise<DashboardData> {
  return fetchWithAuth<DashboardData>(`/api/dashboard/${userId}`);
}

// Voice Reminder API
export async function speakReminder(reminderId: number): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>(`/api/reminders/speak/${reminderId}`, {
    method: "POST",
  });
}

export async function speakCustomMessage(
  message: string, 
  userId?: number, 
  priority: string = "medium", 
  voiceSettings?: any
): Promise<{ success: boolean }> {
  return fetchWithAuth<{ success: boolean }>('/api/speak', {
    method: "POST",
    body: JSON.stringify({
      message,
      user_id: userId,
      priority,
      voice_settings: voiceSettings || {}
    }),
  });
}

export async function getAvailableVoices(): Promise<{ voices: Array<{id: string, name: string, gender: string, engine: string}> }> {
  return fetchWithAuth<{ voices: Array<{id: string, name: string, gender: string, engine: string}> }>('/api/voices');
}

// Create a backend API instance for direct use
export const backendApi = {
  get: async (endpoint: string) => {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          "Accept": "application/json",
        },
        credentials: 'include',
        mode: 'cors',
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || response.statusText);
      }
      
      return {
        data: await response.json(),
        status: response.status,
      };
    } catch (error) {
      console.error(`API Error:`, error);
      throw error;
    }
  },
  
  post: async (endpoint: string, data?: any) => {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: data ? JSON.stringify(data) : undefined,
        credentials: 'include',
        mode: 'cors',
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || response.statusText);
      }
      
      return {
        data: await response.json(),
        status: response.status,
      };
    } catch (error) {
      console.error(`API Error:`, error);
      throw error;
    }
  }
};

// ML Agent API Functions - Use a different name to avoid duplication
export async function mlFetch(endpoint: string, options?: RequestInit) {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        ...options?.headers,
      },
      credentials: 'include',
      mode: 'cors',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || response.statusText);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Error:`, error);
    throw error;
  }
}

// ML Insights API
export async function getMLInsights(userId: string) {
  return mlFetch(`/api/ml-insights/${userId}`);
}

export async function getMLModelStatus() {
  return mlFetch('/api/ml-models/status');
}

// Agent Trigger APIs
export async function triggerHealthAgent(userId: string): Promise<{ success: boolean, results?: any, message?: string }> {
  try {
    const response = await mlFetch(`/api/users/${userId}/trigger-health-agent`, { 
      method: 'POST' 
    });
    console.log("Health agent trigger response:", response);
    return {
      success: response.success,
      results: response.results || null,
      message: response.message
    };
  } catch (error) {
    console.error("Error triggering health agent:", error);
    return { 
      success: false, 
      message: error instanceof Error ? error.message : 'An error occurred' 
    };
  }
}

export async function triggerSafetyAgent(userId: string): Promise<{ success: boolean, results?: any, message?: string }> {
  try {
    const response = await mlFetch(`/api/users/${userId}/trigger-safety-agent`, { 
      method: 'POST' 
    });
    console.log("Safety agent trigger response:", response);
    return {
      success: response.success,
      results: response.results || null,
      message: response.message
    };
  } catch (error) {
    console.error("Error triggering safety agent:", error);
    return { 
      success: false, 
      message: error instanceof Error ? error.message : 'An error occurred' 
    };
  }
}

export async function triggerReminderAgent(userId: string) {
  return mlFetch(`/api/users/${userId}/trigger-reminder-agent`, { method: 'POST' });
}

export async function optimizeReminders(userId: string) {
  return mlFetch(`/api/users/${userId}/optimize-reminders`, { method: 'POST' });
}

export async function getHealthPredictions(userId: string) {
  return mlFetch(`/api/users/${userId}/health-predictions`);
}

export async function getSafetyRiskAssessment(userId: string) {
  return mlFetch(`/api/users/${userId}/safety-risk-assessment`);
}

export async function getReminderEffectivenessReport(userId: string) {
  return mlFetch(`/api/users/${userId}/reminder-effectiveness`);
}
