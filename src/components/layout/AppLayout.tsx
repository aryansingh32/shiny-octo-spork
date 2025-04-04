import React, { useEffect, useState } from "react";
import { 
  Sidebar, 
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
  SidebarTrigger
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { 
  Users, 
  Bell, 
  Clock, 
  Activity, 
  Settings, 
  Home, 
  Sun, 
  Moon, 
  Mic 
} from "lucide-react";
import { useTheme } from "@/context/ThemeContext";
import { useVoice } from "@/context/VoiceContext";
import { Link, useLocation } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { useQuery } from "@tanstack/react-query";
import { getActiveAlerts, getAlertsCount } from "@/services/backendApi";

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const location = useLocation();
  const { theme, setTheme } = useTheme();
  const { voiceEnabled, toggleVoice } = useVoice();
  
  // Fetch active alerts
  const { data: alerts } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => getActiveAlerts(),
    refetchInterval: 60000, // Refetch every minute
  });

  // Fetch alerts count - this is more efficient than filtering the alerts
  const { data: alertsCount } = useQuery({
    queryKey: ["alerts-count"],
    queryFn: () => getAlertsCount(),
    refetchInterval: 60000, // Refetch every minute
  });
  
  // Number of active (non-acknowledged) alerts
  const activeAlerts = alerts?.filter(alert => !alert.acknowledged)?.length || 
                      alertsCount?.total || 
                      0;

  const menuItems = [
    { title: "Dashboard", icon: Home, path: "/" },
    { title: "Users", icon: Users, path: "/users" },
    { title: "Reminders", icon: Clock, path: "/reminders" },
    { title: "Alerts", icon: Bell, path: "/alerts", badge: activeAlerts },
    { title: "Health", icon: Activity, path: "/health" },
    { title: "Settings", icon: Settings, path: "/settings" },
  ];

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <Sidebar>
        <SidebarHeader className="flex items-center justify-center p-4">
          <div className="flex items-center gap-2">
            <Activity className="h-7 w-7 text-primary" />
            <h1 className="text-xl font-semibold text-primary">ElderCare</h1>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupContent>
              <SidebarMenu>
                {menuItems.map((item) => (
                  <SidebarMenuItem key={item.path}>
                    <SidebarMenuButton asChild>
                      <Link 
                        to={item.path} 
                        className={`relative ${location.pathname === item.path ? "text-primary font-medium" : ""}`}
                      >
                        <item.icon />
                        <span>{item.title}</span>
                        {item.badge && item.badge > 0 && (
                          <Badge className="absolute -right-1 -top-1 bg-destructive animate-pulse-alert">
                            {item.badge}
                          </Badge>
                        )}
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          <div className="flex flex-col gap-2 p-4">
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
              aria-label={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {theme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </Button>
            <Button 
              variant={voiceEnabled ? "default" : "outline"} 
              size="icon"
              onClick={toggleVoice}
              title={voiceEnabled ? "Disable Voice Commands" : "Enable Voice Commands"}
              aria-label={voiceEnabled ? "Disable Voice Commands" : "Enable Voice Commands"}
            >
              <Mic className="h-5 w-5" />
            </Button>
          </div>
        </SidebarFooter>
      </Sidebar>
      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="border-b bg-background p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <SidebarTrigger className="lg:hidden" />
              <h1 className="text-2xl font-semibold">
                {menuItems.find(item => item.path === location.pathname)?.title || "Dashboard"}
              </h1>
            </div>
            <div className="flex items-center gap-2">
              {activeAlerts > 0 && (
                <Button variant="destructive" size="sm" asChild>
                  <Link to="/alerts">
                    <Bell className="mr-2 h-4 w-4" />
                    {activeAlerts} Active {activeAlerts === 1 ? "Alert" : "Alerts"}
                  </Link>
                </Button>
              )}
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-auto p-6">{children}</main>
      </div>
    </div>
  );
};
