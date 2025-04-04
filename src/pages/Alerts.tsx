import React, { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getActiveAlerts, getUsersFromBackend, acknowledgeAlertInBackend } from "@/services/backendApi";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, CheckSquare, AlertCircle, AlertTriangle, User, Calendar, Clock } from "lucide-react";
import { toast } from "sonner";
import { format } from "date-fns";
import { Alert as AlertUI, AlertDescription, AlertTitle } from "@/components/ui/alert";

// Import the Alert interface from api.ts
import { Alert } from "@/services/api";

const Alerts = () => {
  const queryClient = useQueryClient();
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  // Fetch alerts 
  const { data: alerts, isLoading, error } = useQuery({
    queryKey: ["alerts"],
    queryFn: async () => {
      const data = await getActiveAlerts();
      console.log("Raw alerts data:", data); // Log the raw data to see what we're getting
      return data;
    },
  });

  // Fetch users for mapping IDs to names
  const { data: users } = useQuery({
    queryKey: ["users"],
    queryFn: () => getUsersFromBackend(),
  });

  // Function to safely format dates
  const formatDate = (dateValue: string | number | Date | null | undefined): string => {
    if (!dateValue) return "Unknown date";
    
    try {
      // Try direct formatting
      return format(new Date(dateValue), "MMM d, yyyy 'at' h:mm a");
    } catch (e) {
      console.error(`Error formatting date value: ${dateValue}`, e);
      
      // Try alternative approaches if the value is a string
      if (typeof dateValue === 'string') {
        try {
          // If it's a timestamp like "2025-04-04 14:31:15"
          if (dateValue.includes('-') && dateValue.includes(':')) {
            return dateValue.replace('T', ' ');
          }
          
          // If it's a Unix timestamp in seconds (unlikely for our API)
          const numValue = parseInt(dateValue, 10);
          if (!isNaN(numValue)) {
            return format(new Date(numValue * 1000), "MMM d, yyyy 'at' h:mm a");
          }
        } catch {
          // Ignore nested errors and fallback to default
        }
      }
      
      return "Invalid date";
    }
  };

  // Acknowledge alert mutation
  const acknowledgeMutation = useMutation({
    mutationFn: (alertId: string) => acknowledgeAlertInBackend(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      queryClient.invalidateQueries({ queryKey: ["alerts-count"] });
      toast.success("Alert acknowledged successfully");
      setSelectedAlert(null);
    },
    onError: (error) => {
      toast.error(`Error acknowledging alert: ${error instanceof Error ? error.message : String(error)}`);
    },
  });

  const handleAcknowledge = (alertId: string) => {
    acknowledgeMutation.mutate(alertId);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "bg-destructive text-destructive-foreground";
      case "warning":
        return "bg-yellow-500 text-yellow-50";
      case "info":
        return "bg-blue-500 text-blue-50";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "critical":
        return <AlertCircle className="h-5 w-5" />;
      case "warning":
        return <AlertTriangle className="h-5 w-5" />;
      default:
        return <Bell className="h-5 w-5" />;
    }
  };

  const getUserName = (userId: string) => {
    const user = users?.find((user) => user.id === userId);
    return user ? user.name : "Unknown User";
  };

  const getActiveAlertCount = () => {
    return alerts?.filter((alert: Alert) => !alert.acknowledged).length || 0;
  };

  return (
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Alerts</h1>
        <Badge variant="outline" className="text-base py-1 px-3">
          {getActiveAlertCount()} Active Alerts
        </Badge>
      </div>

      {isLoading ? (
        <div className="flex justify-center p-8">
          <p>Loading alerts...</p>
        </div>
      ) : error ? (
        <AlertUI variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load alerts"}
          </AlertDescription>
        </AlertUI>
      ) : alerts && alerts.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {alerts.map((alert: Alert) => (
            <Card
              key={alert.id}
              className={`${
                alert.acknowledged ? "opacity-60" : ""
              } transition-all hover:shadow-md`}
            >
              <CardHeader className="pb-2">
                <div className="flex justify-between items-center">
                  <Badge className={getSeverityColor(alert.severity)}>
                    <div className="flex items-center gap-1">
                      {getSeverityIcon(alert.severity)}
                      <span className="capitalize">{alert.severity} Priority</span>
                    </div>
                  </Badge>
                  <Badge variant={alert.acknowledged ? "outline" : "secondary"}>
                    {alert.acknowledged ? "Acknowledged" : "New"}
                  </Badge>
                </div>
                <CardTitle className="mt-2">{alert.title}</CardTitle>
                <CardDescription className="flex items-center gap-1">
                  <User className="h-3 w-3" />
                  {getUserName(alert.userId)}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm">{alert.description}</p>
                <div className="flex items-center gap-1 mt-3 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>
                    {formatDate(alert.dateTime)}
                  </span>
                </div>
              </CardContent>
              <CardFooter>
                {!alert.acknowledged && (
                  <Button
                    className="w-full"
                    variant="outline"
                    onClick={() => handleAcknowledge(alert.id)}
                    disabled={acknowledgeMutation.isPending}
                  >
                    <CheckSquare className="mr-2 h-4 w-4" />
                    Acknowledge
                  </Button>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="bg-muted/50">
          <CardContent className="flex flex-col items-center justify-center py-10">
            <Bell className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-xl font-medium mb-2">No Active Alerts</h3>
            <p className="text-muted-foreground text-center max-w-md">
              There are currently no alerts that require your attention. Check back later or refresh
              the page.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Alerts;
