import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getHealthAnomalies,
  getUsersFromBackend,
  getHealthData,
  getLatestHealthData,
  submitHealthData,
} from "@/services/backendApi";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, Heart, User, Clock } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { format } from "date-fns";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const Health = () => {
  const queryClient = useQueryClient();
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string>("heartRate");

  // Fetch users
  const { data: users } = useQuery({
    queryKey: ["users"],
    queryFn: getUsersFromBackend,
  });

  // Fetch health anomalies
  const { data: anomalies, isLoading: anomaliesLoading, error: anomaliesError } = useQuery({
    queryKey: ["healthAnomalies"],
    queryFn: getHealthAnomalies,
  });

  // Fetch user's health data
  const {
    data: healthData,
    isLoading: healthDataLoading,
    error: healthDataError,
  } = useQuery({
    queryKey: ["healthData", selectedUser, selectedMetric],
    queryFn: () => (selectedUser ? getHealthData(selectedUser, selectedMetric) : Promise.resolve([])),
    enabled: !!selectedUser,
  });

  // Fetch user's latest health data
  const {
    data: latestHealthData,
    isLoading: latestHealthDataLoading,
    error: latestHealthDataError,
  } = useQuery({
    queryKey: ["latestHealthData", selectedUser],
    queryFn: () => (selectedUser ? getLatestHealthData(selectedUser) : Promise.resolve({})),
    enabled: !!selectedUser,
  });

  // Submit health data mutation
  const submitHealthDataMutation = useMutation({
    mutationFn: ({ userId, data }: { userId: string; data: any }) => submitHealthData(userId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["healthData"] });
      queryClient.invalidateQueries({ queryKey: ["latestHealthData"] });
      toast.success("Health data submitted successfully");
    },
    onError: (error) => {
      toast.error(`Failed to submit health data: ${error instanceof Error ? error.message : "Unknown error"}`);
    },
  });

  const handleUserChange = (userId: string) => {
    setSelectedUser(userId);
  };

  const handleMetricChange = (metric: string) => {
    setSelectedMetric(metric);
  };

  const getUserName = (userId: string) => {
    const user = users?.find((user) => user.id === userId);
    return user ? user.name : "Unknown User";
  };

  const formatHealthData = (data: any[]) => {
    if (!data || data.length === 0) return [];

    return data.map((item) => ({
      date: format(new Date(item.dateTime), "MM/dd"),
      value: item.type === "bloodPressure" ? item.value.systolic : item.value,
      diastolic: item.type === "bloodPressure" ? item.value.diastolic : undefined,
    }));
  };

  const getMetricDisplayName = (metric: string) => {
    switch (metric) {
      case "heartRate":
        return "Heart Rate";
      case "bloodPressure":
        return "Blood Pressure";
      case "temperature":
        return "Temperature";
      case "bloodSugar":
        return "Blood Sugar";
      case "oxygenLevel":
        return "Oxygen Level";
      case "weight":
        return "Weight";
      case "sleep":
        return "Sleep";
      default:
        return metric;
    }
  };

  const getMetricUnit = (metric: string) => {
    switch (metric) {
      case "heartRate":
        return "bpm";
      case "bloodPressure":
        return "mmHg";
      case "temperature":
        return "°C";
      case "bloodSugar":
        return "mg/dL";
      case "oxygenLevel":
        return "%";
      case "weight":
        return "kg";
      case "sleep":
        return "hours";
      default:
        return "";
    }
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case "heartRate":
        return <Heart className="h-4 w-4 text-red-500" />;
      case "sleep":
        return <Clock className="h-4 w-4 text-blue-500" />;
      default:
        return <AlertCircle className="h-4 w-4" />;
    }
  };

  const healthMetrics = [
    "heartRate",
    "bloodPressure",
    "temperature",
    "bloodSugar",
    "oxygenLevel",
    "weight",
    "sleep",
  ];

  // Get color for abnormal values
  const getAbnormalColor = (isAbnormal: boolean | undefined) => {
    return isAbnormal ? "text-red-500" : "text-green-500";
  };

  const formatTime = (timestamp: string) => {
    try {
      if (!timestamp) return 'No data';
      const date = new Date(timestamp);
      if (isNaN(date.getTime())) return 'Invalid date';
      return format(date, 'MMM d, yyyy HH:mm');
    } catch (error) {
      return 'Invalid date';
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Health Monitoring</h1>

      <div className="mb-6">
        <Card>
          <CardHeader>
            <CardTitle>User Selection</CardTitle>
            <CardDescription>Select a user to view their health data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="w-full md:w-1/2">
                <Select onValueChange={handleUserChange}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a user" />
                  </SelectTrigger>
                  <SelectContent>
                    {users?.map((user) => (
                      <SelectItem key={user.id} value={user.id}>
                        {user.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {selectedUser && (
                <div className="w-full md:w-1/2">
                  <Select value={selectedMetric} onValueChange={handleMetricChange}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select health metric" />
                    </SelectTrigger>
                    <SelectContent>
                      {healthMetrics.map((metric) => (
                        <SelectItem key={metric} value={metric}>
                          {getMetricDisplayName(metric)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {selectedUser ? (
        <>
          <Tabs defaultValue="dashboard">
            <TabsList className="mb-4">
              <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
              <TabsTrigger value="trends">Trends</TabsTrigger>
              <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
            </TabsList>

            <TabsContent value="dashboard">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                {latestHealthDataLoading ? (
                  <p>Loading latest health data...</p>
                ) : latestHealthDataError ? (
                  <Card className="border-destructive col-span-full">
                    <CardHeader>
                      <div className="flex items-center">
                        <AlertCircle className="h-6 w-6 text-destructive mr-2" />
                        <CardTitle>Error Loading Health Data</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p>
                        {latestHealthDataError instanceof Error
                          ? latestHealthDataError.message
                          : "Failed to load health data"}
                      </p>
                    </CardContent>
                  </Card>
                ) : (
                  Object.entries(latestHealthData || {}).map(([key, data]: [string, any]) => (
                    <Card key={key}>
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm font-medium">
                            {getMetricDisplayName(key)}
                          </CardTitle>
                          <div className="flex items-center">
                            {getMetricIcon(key)}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">
                          {key === "bloodPressure" ? (
                            <span>
                              <span className={getAbnormalColor(data.isAbnormal)}>
                                {data.value.systolic}/{data.value.diastolic}
                              </span>{" "}
                              {getMetricUnit(key)}
                            </span>
                          ) : (
                            <span>
                              <span className={getAbnormalColor(data.isAbnormal)}>
                                {key === "temperature" ? data.value.toFixed(1) : data.value}
                              </span>{" "}
                              {getMetricUnit(key)}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Last updated: {formatTime(data.dateTime)}
                        </p>
                        {data.isAbnormal && (
                          <Badge variant="destructive" className="mt-2">
                            Abnormal
                          </Badge>
                        )}
                      </CardContent>
                    </Card>
                  ))
                )}
              </div>
            </TabsContent>

            <TabsContent value="trends">
              <Card>
                <CardHeader>
                  <CardTitle>
                    {getMetricDisplayName(selectedMetric)} Trends for {getUserName(selectedUser)}
                  </CardTitle>
                  <CardDescription>
                    View {getMetricDisplayName(selectedMetric).toLowerCase()} trends over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {healthDataLoading ? (
                    <div className="h-64 flex items-center justify-center">
                      <p>Loading health data...</p>
                    </div>
                  ) : healthDataError ? (
                    <div>
                      <p className="text-red-500">
                        {healthDataError instanceof Error
                          ? healthDataError.message
                          : "Failed to load health data"}
                      </p>
                    </div>
                  ) : healthData && healthData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart
                        data={formatHealthData(healthData)}
                        margin={{
                          top: 5,
                          right: 30,
                          left: 20,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="value"
                          name={
                            selectedMetric === "bloodPressure"
                              ? "Systolic"
                              : getMetricDisplayName(selectedMetric)
                          }
                          stroke="#8884d8"
                          activeDot={{ r: 8 }}
                        />
                        {selectedMetric === "bloodPressure" && (
                          <Line
                            type="monotone"
                            dataKey="diastolic"
                            name="Diastolic"
                            stroke="#82ca9d"
                          />
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-64 flex items-center justify-center">
                      <p>No {selectedMetric} data available for this user</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="anomalies">
              <Card>
                <CardHeader>
                  <CardTitle>Health Anomalies</CardTitle>
                  <CardDescription>
                    Review detected health anomalies that require attention
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {anomaliesLoading ? (
                    <div className="h-64 flex items-center justify-center">
                      <p>Loading anomalies...</p>
                    </div>
                  ) : anomaliesError ? (
                    <div>
                      <p className="text-red-500">
                        {anomaliesError instanceof Error
                          ? anomaliesError.message
                          : "Failed to load anomalies"}
                      </p>
                    </div>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>User</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>Metric</TableHead>
                          <TableHead>Value</TableHead>
                          <TableHead>Expected Range</TableHead>
                          <TableHead>Time</TableHead>
                          <TableHead>Severity</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {anomalies && anomalies.length > 0 ? (
                          anomalies
                            .filter(
                              (anomaly) => selectedUser ? anomaly.userId === selectedUser : true
                            )
                            .map((anomaly) => (
                              <TableRow key={anomaly.id}>
                                <TableCell className="font-medium">
                                  <div className="flex items-center">
                                    <User className="h-4 w-4 mr-2" />
                                    {getUserName(anomaly.userId)}
                                  </div>
                                </TableCell>
                                <TableCell>{anomaly.type}</TableCell>
                                <TableCell>{anomaly.metric}</TableCell>
                                <TableCell className="font-semibold text-red-500">{
                                  typeof anomaly.value === 'object'
                                    ? `${anomaly.value.systolic}/${anomaly.value.diastolic}`
                                    : anomaly.value
                                }</TableCell>
                                <TableCell>
                                  {anomaly.expectedRange.min} - {anomaly.expectedRange.max}
                                </TableCell>
                                <TableCell>
                                  {formatTime(anomaly.timestamp)}
                                </TableCell>
                                <TableCell>
                                  <Badge
                                    variant={
                                      anomaly.severity === "high"
                                        ? "destructive"
                                        : anomaly.severity === "medium"
                                        ? "default"
                                        : "secondary"
                                    }
                                  >
                                    {anomaly.severity}
                                  </Badge>
                                </TableCell>
                              </TableRow>
                            ))
                        ) : (
                          <TableRow>
                            <TableCell colSpan={7} className="text-center">
                              No health anomalies found for this user. That's good news!
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      ) : (
        <Card className="bg-muted/50">
          <CardContent className="flex flex-col items-center justify-center py-10">
            <User className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-xl font-medium mb-2">No User Selected</h3>
            <p className="text-muted-foreground text-center max-w-md">
              Please select a user from the dropdown menu above to view their health monitoring data.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Health;
