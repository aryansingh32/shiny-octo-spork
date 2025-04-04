
import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { useQuery } from "@tanstack/react-query";
import { checkBackendHealth } from "@/services/backendApi";
import { AlertCircle, Check, Server, Shield, UserCog, Bell, Clock } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const Settings = () => {
  const [apiUrl, setApiUrl] = useState("/api");
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [smsNotifications, setSmsNotifications] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Check backend health
  const { data: healthStatus, isLoading: healthLoading, error: healthError, refetch: refetchHealth } = useQuery({
    queryKey: ["backendHealth"],
    queryFn: checkBackendHealth,
    retry: 1,
    refetchInterval: autoRefresh ? 60000 : false, // Refresh every minute if autoRefresh is true
  });

  const handleSaveApiSettings = () => {
    // In a real implementation, you would save these settings to localStorage or a backend
    toast.success("API settings saved successfully");
    refetchHealth();
  };

  const handleSaveNotificationSettings = () => {
    toast.success("Notification settings saved successfully");
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Settings</h1>

      <Tabs defaultValue="general">
        <TabsList className="mb-6">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="api">API & Backend</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>Configure general application settings</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="auto-refresh" className="font-medium">
                      Auto-refresh dashboard
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically refresh data on the dashboard
                    </p>
                  </div>
                  <Switch
                    id="auto-refresh"
                    checked={autoRefresh}
                    onCheckedChange={setAutoRefresh}
                  />
                </div>

                <Separator />

                <div>
                  <Label className="font-medium">Theme</Label>
                  <p className="text-sm text-muted-foreground mb-2">
                    Select your preferred theme
                  </p>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" className="px-4">Light</Button>
                    <Button variant="default" size="sm" className="px-4">Dark</Button>
                    <Button variant="outline" size="sm" className="px-4">System</Button>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button onClick={() => toast.success("Settings saved")}>Save settings</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="api">
          <Card>
            <CardHeader>
              <CardTitle>API & Backend Settings</CardTitle>
              <CardDescription>Configure your backend API connection</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <Label htmlFor="api-url">API Base URL</Label>
                  <Input
                    id="api-url"
                    placeholder="http://localhost:5000/api"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className="mt-1"
                  />
                  <p className="text-sm text-muted-foreground mt-2">
                    The base URL for your backend API. Default is "/api" for same-origin deployment.
                  </p>
                </div>

                <div className="bg-muted/40 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium flex items-center">
                      <Server className="h-4 w-4 mr-2" />
                      Backend Health Status
                    </h3>
                    <Button variant="outline" size="sm" onClick={() => refetchHealth()}>
                      Refresh
                    </Button>
                  </div>

                  {healthLoading ? (
                    <p>Checking backend health...</p>
                  ) : healthError ? (
                    <div className="flex items-center text-destructive">
                      <AlertCircle className="h-4 w-4 mr-2" />
                      <span>
                        Backend not available. Please check your connection.
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center text-green-600">
                      <Check className="h-4 w-4 mr-2" />
                      <span>
                        Backend is operational{" "}
                        <Badge variant="outline" className="bg-green-100 ml-2">
                          {healthStatus?.status || "OK"}
                        </Badge>
                      </span>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <h3 className="font-medium">Available Endpoints</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="border rounded-md p-3">
                      <div className="flex items-center">
                        <UserCog className="h-4 w-4 mr-2" />
                        <span className="font-medium">User Management</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        /api/users
                      </p>
                    </div>
                    <div className="border rounded-md p-3">
                      <div className="flex items-center">
                        <Bell className="h-4 w-4 mr-2" />
                        <span className="font-medium">Alert Management</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        /api/alerts
                      </p>
                    </div>
                    <div className="border rounded-md p-3">
                      <div className="flex items-center">
                        <Clock className="h-4 w-4 mr-2" />
                        <span className="font-medium">Reminders</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        /api/reminders
                      </p>
                    </div>
                    <div className="border rounded-md p-3">
                      <div className="flex items-center">
                        <Shield className="h-4 w-4 mr-2" />
                        <span className="font-medium">Safety</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        /api/safety
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button onClick={handleSaveApiSettings}>Save API settings</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle>Notification Settings</CardTitle>
              <CardDescription>Configure how you receive notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="notifications-enabled" className="font-medium">
                      Enable notifications
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Receive notifications about alerts and reminders
                    </p>
                  </div>
                  <Switch
                    id="notifications-enabled"
                    checked={notificationsEnabled}
                    onCheckedChange={setNotificationsEnabled}
                  />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="email-notifications" className="font-medium">
                      Email notifications
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Receive notifications via email
                    </p>
                  </div>
                  <Switch
                    id="email-notifications"
                    checked={emailNotifications}
                    disabled={!notificationsEnabled}
                    onCheckedChange={setEmailNotifications}
                  />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="sms-notifications" className="font-medium">
                      SMS notifications
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Receive notifications via SMS
                    </p>
                  </div>
                  <Switch
                    id="sms-notifications"
                    checked={smsNotifications}
                    disabled={!notificationsEnabled}
                    onCheckedChange={setSmsNotifications}
                  />
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button onClick={handleSaveNotificationSettings}>Save notification settings</Button>
            </CardFooter>
          </Card>
        </TabsContent>

        <TabsContent value="system">
          <Card>
            <CardHeader>
              <CardTitle>System Settings</CardTitle>
              <CardDescription>Advanced system configuration</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="font-medium">System Information</h3>
                  <div className="border rounded-md p-4 mt-2 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Application Version</span>
                      <span className="text-sm font-medium">1.0.0</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Backend Version</span>
                      <span className="text-sm font-medium">
                        {healthStatus ? "1.0.0" : "Unknown"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Last Updated</span>
                      <span className="text-sm font-medium">April 3, 2025</span>
                    </div>
                  </div>
                </div>

                <Separator />

                <div>
                  <h3 className="font-medium">Diagnostic Tools</h3>
                  <div className="flex gap-2 mt-2">
                    <Button variant="outline" size="sm" onClick={() => refetchHealth()}>
                      Test API Connection
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => toast.success("Cache cleared")}>
                      Clear Cache
                    </Button>
                  </div>
                </div>

                <Separator />

                <div>
                  <h3 className="font-medium">Danger Zone</h3>
                  <div className="border border-destructive rounded-md p-4 mt-2">
                    <p className="text-sm mb-4">
                      These actions cannot be undone. Please proceed with caution.
                    </p>
                    <div className="flex gap-2">
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => {
                          if (window.confirm("Are you sure? This will reset all application data.")) {
                            toast.success("Application data has been reset");
                          }
                        }}
                      >
                        Reset All Data
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Settings;
