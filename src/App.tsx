import React from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/context/ThemeContext";
import { VoiceProvider } from "@/context/VoiceContext";
import Dashboard from "@/pages/Dashboard";
import NotFound from "./pages/NotFound";
import { AppLayout } from "@/components/layout/AppLayout";
import { SidebarProvider } from "@/components/ui/sidebar";
import Users from "@/pages/Users";
import Reminders from "@/pages/Reminders";
import Alerts from "@/pages/Alerts";
import Health from "@/pages/Health";
import Settings from "@/pages/Settings";
import Safety from "./pages/Safety";

// Create a client
const queryClient = new QueryClient();

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <VoiceProvider>
          <TooltipProvider>
            <SidebarProvider>
              <Toaster />
              <Sonner />
              <BrowserRouter>
                <Routes>
                  <Route path="/" element={
                    <AppLayout>
                      <Dashboard />
                    </AppLayout>
                  } />
                  <Route path="/users" element={
                    <AppLayout>
                      <Users />
                    </AppLayout>
                  } />
                  <Route path="/reminders" element={
                    <AppLayout>
                      <Reminders />
                    </AppLayout>
                  } />
                  <Route path="/alerts" element={
                    <AppLayout>
                      <Alerts />
                    </AppLayout>
                  } />
                  <Route path="/health" element={
                    <AppLayout>
                      <Health />
                    </AppLayout>
                  } />
                  <Route path="/safety" element={
                    <AppLayout>
                      <Safety />
                    </AppLayout>
                  } />
                  <Route path="/settings" element={
                    <AppLayout>
                      <Settings />
                    </AppLayout>
                  } />
                  {/* Catch-all route */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </BrowserRouter>
            </SidebarProvider>
          </TooltipProvider>
        </VoiceProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
