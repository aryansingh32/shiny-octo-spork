import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useQuery } from "@tanstack/react-query";
import { format } from "date-fns";
import { Shield, AlertCircle, MapPin, Activity, Loader2 } from "lucide-react";
import { getUsersFromBackend, getDashboardData } from "@/services/backendApi";
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
} from "recharts";

const Safety = () => {
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);

  // Fetch users to populate the dropdown
  const { data: users = [], isLoading: usersLoading } = useQuery({
    queryKey: ["users"],
    queryFn: () => getUsersFromBackend(),
  });

  // Fetch safety data for the selected user
  const { data: dashboardData, isLoading: isDashboardLoading, error: dashboardError } = useQuery({
    queryKey: ['dashboardData', selectedUserId],
    queryFn: () => getDashboardData(selectedUserId),
    enabled: !!selectedUserId,
  });

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Safety Monitoring</h1>

      <div className="mb-6">
        <Card>
          <CardHeader>
            <CardTitle>User Selection</CardTitle>
            <CardDescription>Select a user to view their safety data</CardDescription>
          </CardHeader>
          <CardContent>
            <Select onValueChange={(value) => setSelectedUserId(value)}>
              <SelectTrigger>
                <SelectValue placeholder="Select a user" />
              </SelectTrigger>
              <SelectContent>
                {users.map((user) => (
                  <SelectItem key={user.id} value={user.id.toString()}>
                    {user.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
      </div>

      {selectedUserId && (
        <div>
          {isDashboardLoading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : dashboardError ? (
            <div className="flex items-center justify-center h-64">
              <AlertCircle className="h-8 w-8 text-red-500" />
              <p className="ml-2 text-red-500">Error loading safety data</p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Current Status</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {dashboardData?.safety_status?.status || 'Unknown'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Last update: {dashboardData?.safety_status?.last_reading || 'No data'}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Location</CardTitle>
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {dashboardData?.safety_status?.location || 'Unknown'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Current location
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Movement</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {dashboardData?.safety_status?.movement_type || 'Unknown'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Movement type
                  </p>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Safety; 