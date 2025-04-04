import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useQuery } from "@tanstack/react-query";
import { format } from "date-fns";
import { 
  Activity, 
  AlertCircle, 
  Bell, 
  ClipboardList, 
  Clock, 
  Heart, 
  Home, 
  ShieldAlert, 
  Thermometer, 
  User, 
  Users,
  Loader2,
  ListTodo,
  Shield
} from "lucide-react";
import { 
  getDashboardData, 
  getActiveAlerts, 
  getUsersFromBackend, 
  getRemindersFromBackend,
  getUsersCount,
  getRemindersCount,
  getAlertsCount
} from "@/services/backendApi";
import { Link } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import VoiceReminderWidget from "@/components/VoiceReminderWidget";
import MLInsightsPanel from "@/components/MLInsightsPanel";
import AgentControl from "@/components/AgentControl";
import DataSubmissionForm from "@/components/DataSubmissionForm";

const getUnitForMetric = (metricName: string): string => {
  const units: Record<string, string> = {
    heart_rate: 'bpm',
    blood_pressure: 'mmHg',
    temperature: '°C',
    oxygen_saturation: '%',
    glucose_level: 'mg/dL',
    weight: 'kg',
    height: 'cm',
    bmi: 'kg/m²'
  };
  return units[metricName] || '';
};

const Dashboard = () => {
  const [selectedUser, setSelectedUser] = useState<string | null>(null);

  // Fetch users to populate the dropdown
  const { data: users = [], isLoading: usersLoading } = useQuery({
    queryKey: ["users"],
    queryFn: () => getUsersFromBackend(),
  });

  // Fetch dashboard data for the selected user
  const { data: dashboardData, isLoading: isDashboardLoading, error: dashboardError } = useQuery({
    queryKey: ['dashboardData', selectedUser],
    queryFn: () => getDashboardData(selectedUser),
    enabled: !!selectedUser,
  });

  // Fetch aggregate counts for the overview
  const { data: usersCount } = useQuery({
    queryKey: ["users-count"],
    queryFn: () => getUsersCount(),
  });

  const { data: remindersCount, refetch: refetchReminderCount } = useQuery({
    queryKey: ["reminders-count"],
    queryFn: () => getRemindersCount(),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const { data: alertsCount = { total: 0, byType: {}, bySeverity: { low: 0, medium: 0, high: 0 } }, isLoading: alertsCountLoading } = useQuery({
    queryKey: ['alertsCount'],
    queryFn: () => getAlertsCount(),
  });

  // Get active alerts
  const { data: alerts = [], isLoading: alertsLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => getActiveAlerts(),
  });

  const { data: reminders } = useQuery({
    queryKey: ["reminders"],
    queryFn: () => getRemindersFromBackend(),
  });

  // Fix the alerts filtering
  const unacknowledgedAlerts = Array.isArray(alerts) ? alerts.filter(alert => !alert.acknowledged) : [];
  const highPriorityAlerts = Array.isArray(alerts) ? alerts.filter(alert => alert.severity === 'high' && !alert.acknowledged) : [];

  const alertsByType = alertsCount?.byType || {};
  const alertTypes = Object.keys(alertsByType);
  const alertData = alertTypes.map(type => ({ name: type, value: alertsByType[type] }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A28DFF'];

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "alert":
        return "text-red-500";
      case "caution":
        return "text-yellow-500";
      case "normal":
        return "text-green-500";
      default:
        return "";
    }
  };

  // Get trend icon
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving":
        return <div className="text-green-500">↑</div>;
      case "declining":
        return <div className="text-red-500">↓</div>;
      default:
        return <div className="text-gray-500">→</div>;
    }
  };

  // Format metrics for the chart
  const formatMetricsForChart = (metrics: any) => {
    if (!metrics) return [];
    
    return Object.keys(metrics).map(key => ({
      name: key,
      value: metrics[key].current,
      unit: metrics[key].unit,
    }));
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Total Users Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-between items-center">
              <div className="text-2xl font-bold">{usersCount?.total || users.length || 0}</div>
              <Users className="h-5 w-5 text-blue-500" />
            </div>
          </CardContent>
          <CardFooter className="pt-0">
            <Button variant="outline" size="sm" asChild className="w-full">
              <Link to="/users">
                View All Users
              </Link>
            </Button>
          </CardFooter>
        </Card>

        {/* Active Reminders Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Upcoming Reminders</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-between items-center">
              <div className="text-2xl font-bold">{remindersCount?.upcoming || 0}</div>
              <Clock className="h-5 w-5 text-indigo-500" />
            </div>
          </CardContent>
          <CardFooter className="pt-0">
            <Button variant="outline" size="sm" asChild className="w-full">
              <Link to="/reminders">
                View Reminders
              </Link>
            </Button>
          </CardFooter>
        </Card>

        {/* Active Alerts Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-between items-center">
              <div className="text-2xl font-bold">{alertsCount?.total || unacknowledgedAlerts.length || 0}</div>
              <Bell className="h-5 w-5 text-red-500" />
            </div>
          </CardContent>
          <CardFooter className="pt-0">
            <Button variant="outline" size="sm" asChild className="w-full">
              <Link to="/alerts">
                View Alerts
              </Link>
            </Button>
          </CardFooter>
        </Card>
      </div>

      <div className="mb-6">
        <Card>
          <CardHeader>
            <CardTitle>User Dashboard</CardTitle>
            <CardDescription>Select a user to view their detailed health and safety data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="w-full md:w-1/2">
                <Select onValueChange={(value) => setSelectedUser(value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a user" />
                  </SelectTrigger>
                  <SelectContent>
                    {users.map((user) => (
                      <SelectItem key={user.id} value={user.id}>
                        {user.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {selectedUser && (
        <div>
          {isDashboardLoading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : dashboardError ? (
            <div className="flex items-center justify-center h-64">
              <Alert variant="destructive">
                <AlertTitle>Error</AlertTitle>
              <AlertDescription>
                  {dashboardError instanceof Error ? dashboardError.message : 'Failed to load dashboard data'}
              </AlertDescription>
            </Alert>
            </div>
          ) : (
            <Tabs defaultValue="overview">
              <TabsList className="mb-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="health">Health</TabsTrigger>
                <TabsTrigger value="safety">Safety</TabsTrigger>
                <TabsTrigger value="reminders">Reminders</TabsTrigger>
              </TabsList>

              <TabsContent value="overview">
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Health Status</CardTitle>
                      <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {dashboardData?.health_overview?.status || 'Unknown'}
                            </div>
                      <p className="text-xs text-muted-foreground">
                        Last reading: {dashboardData?.health_overview?.last_reading || 'No data'}
                      </p>
                      {dashboardData?.health_overview?.metrics && (
                        <div className="mt-2 space-y-1">
                          {Object.entries(dashboardData.health_overview.metrics).map(([key, value]) => (
                            <div key={key} className="flex justify-between text-sm">
                              <span className="capitalize">{key.replace('_', ' ')}</span>
                              <span>{value || 'N/A'}</span>
                          </div>
                        ))}
                      </div>
                      )}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Safety Status</CardTitle>
                      <Shield className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {dashboardData?.safety_status?.status || 'Unknown'}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Last update: {dashboardData?.safety_status?.last_reading || 'No data'}
                      </p>
                      {dashboardData?.safety_status && (
                        <div className="mt-2 space-y-1">
                          <div className="flex justify-between text-sm">
                            <span>Location</span>
                            <span>{dashboardData.safety_status.location || 'Unknown'}</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span>Movement</span>
                            <span>{dashboardData.safety_status.movement_type || 'Unknown'}</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span>Fall Detected</span>
                            <span>{dashboardData.safety_status.fall_detected ? 'Yes' : 'No'}</span>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Reminders</CardTitle>
                      <Bell className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {dashboardData?.reminder_stats?.total_reminders || 0}
                        </div>
                      <p className="text-xs text-muted-foreground">
                        {dashboardData?.reminder_stats?.acknowledged || 0} acknowledged, {dashboardData?.reminder_stats?.missed || 0} missed
                      </p>
                      {dashboardData?.reminder_stats && (
                        <div className="mt-2 space-y-1">
                          <div className="flex justify-between text-sm">
                            <span>Acknowledgement Rate</span>
                            <span>{dashboardData.reminder_stats.acknowledgement_rate?.toFixed(1) || '0'}%</span>
                        </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Pending Items</CardTitle>
                      <ListTodo className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">
                        {dashboardData?.pending_items?.length || 0}
                        </div>
                      <p className="text-xs text-muted-foreground">
                        Items requiring attention
                      </p>
                      {dashboardData?.pending_items && dashboardData.pending_items.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {dashboardData.pending_items.slice(0, 3).map((item) => (
                            <div key={item.id} className="text-sm">
                              {item.type === 'reminder' ? (
                                <span className="text-muted-foreground">{item.title}</span>
                              ) : (
                                <span className="text-muted-foreground">{item.message}</span>
                              )}
                        </div>
                          ))}
                          {dashboardData.pending_items.length > 3 && (
                            <div className="text-xs text-muted-foreground">
                              +{dashboardData.pending_items.length - 3} more items
                        </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="health">
                <Card>
                  <CardHeader>
                    <CardTitle>Health Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {dashboardData?.health_overview?.metrics ? (
                    <div className="h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart
                            data={Object.entries(dashboardData.health_overview.metrics).map(([name, value]) => ({
                              name: name.replace('_', ' '),
                              value: value || 0,
                              unit: getUnitForMetric(name)
                            }))}
                          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip formatter={(value, name, props) => [`${value} ${props.payload.unit}`, name]} />
                          <Area
                            type="monotone"
                            dataKey="value"
                            stroke="#8884d8"
                            fill="#8884d8"
                            fillOpacity={0.3}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                    ) : (
                      <div className="h-[300px] flex items-center justify-center">
                        <p className="text-muted-foreground">No health metrics data available</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button asChild className="w-full">
                      <Link to="/health">View Detailed Health Data</Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>

              <TabsContent value="safety">
                <Card>
                  <CardHeader>
                    <CardTitle>Safety Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {dashboardData?.safety_status ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                            <p className="text-sm font-medium">Current Status</p>
                            <p className="text-2xl font-bold">{dashboardData.safety_status.status}</p>
                          </div>
                          <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
                            <Shield className="h-6 w-6 text-green-600" />
                          </div>
                        </div>
                        <div className="grid gap-4">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Location</span>
                            <span className="font-medium">{dashboardData.safety_status.location}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Movement</span>
                            <span className="font-medium">{dashboardData.safety_status.movement_type}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Fall Detected</span>
                            <span className="font-medium">{dashboardData.safety_status.fall_detected ? 'Yes' : 'No'}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Last Update</span>
                            <span className="font-medium">{dashboardData.safety_status.last_reading}</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="h-[300px] flex items-center justify-center">
                        <p className="text-muted-foreground">No safety data available</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button asChild className="w-full">
                      <Link to="/safety">View Detailed Safety Data</Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>

              <TabsContent value="reminders">
                <Card>
                  <CardHeader>
                    <CardTitle>Reminders Overview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {dashboardData?.reminders_summary ? (
                    <div className="h-[300px] flex items-center justify-center">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={[
                                { name: "Upcoming", value: dashboardData.reminders_summary.upcoming || 0 },
                                { name: "Missed", value: dashboardData.reminders_summary.missed || 0 },
                                { name: "Completed", value: dashboardData.reminders_summary.completed || 0 }
                            ]}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            <Cell key="upcoming" fill="#3b82f6" />
                            <Cell key="missed" fill="#ef4444" />
                            <Cell key="completed" fill="#22c55e" />
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    ) : (
                      <div className="h-[300px] flex items-center justify-center">
                        <p className="text-muted-foreground">No reminders data available</p>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button asChild className="w-full">
                      <Link to="/reminders">View All Reminders</Link>
                    </Button>
                  </CardFooter>
                </Card>
              </TabsContent>
            </Tabs>
          )}
        </div>
      )}

      {!selectedUser && (
        <div className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>System Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="h-[300px] flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={alertData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {alertData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                <div className="space-y-4">
                  <div className="bg-muted p-4 rounded-md">
                    <div className="flex items-center gap-2 mb-2">
                      <Users className="h-5 w-5 text-blue-500" />
                      <h3 className="font-medium">Users</h3>
                    </div>
                    <p className="text-2xl font-bold">{usersCount?.total || users.length || 0}</p>
                    <p className="text-sm text-muted-foreground">Total registered users</p>
                  </div>

                  <div className="bg-muted p-4 rounded-md">
                    <div className="flex items-center gap-2 mb-2">
                      <ClipboardList className="h-5 w-5 text-indigo-500" />
                      <h3 className="font-medium">Reminders</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <p className="text-xl font-bold">{remindersCount?.upcoming || 0}</p>
                        <p className="text-xs text-muted-foreground">Upcoming</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold">{remindersCount?.missed || 0}</p>
                        <p className="text-xs text-muted-foreground">Missed</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold">{remindersCount?.completed || 0}</p>
                        <p className="text-xs text-muted-foreground">Completed</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-muted p-4 rounded-md">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle className="h-5 w-5 text-red-500" />
                      <h3 className="font-medium">Alerts</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <p className="text-xl font-bold text-red-500">{alertsCount?.bySeverity?.high || 0}</p>
                        <p className="text-xs text-muted-foreground">High</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-yellow-500">{alertsCount?.bySeverity?.medium || 0}</p>
                        <p className="text-xs text-muted-foreground">Medium</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-blue-500">{alertsCount?.bySeverity?.low || 0}</p>
                        <p className="text-xs text-muted-foreground">Low</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {!selectedUser && (
        <div className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Broadcast Voice Reminder</CardTitle>
              <CardDescription>Send a voice reminder to all users</CardDescription>
            </CardHeader>
            <CardContent>
              <VoiceReminderWidget />
            </CardContent>
          </Card>
        </div>
      )}

      {selectedUser && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
          <MLInsightsPanel userId={selectedUser} />
          <div className="space-y-6">
            <AgentControl userId={selectedUser} />
            <DataSubmissionForm userId={selectedUser} />
          </div>
          <div className="md:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Voice Reminder Widget</CardTitle>
                <CardDescription>Send voice reminders to the selected user</CardDescription>
              </CardHeader>
              <CardContent>
                <VoiceReminderWidget userId={selectedUser} />
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
