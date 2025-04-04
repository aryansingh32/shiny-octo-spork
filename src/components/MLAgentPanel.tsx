import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Brain, Activity, Shield, Bell, Loader2 } from 'lucide-react';
import { triggerHealthAgent, triggerSafetyAgent, triggerReminderAgent } from '@/services/backendApi';
import { toast } from 'sonner';
import AgentResultDisplay from './AgentResultDisplay';

interface MLAgentPanelProps {
  userId?: string;
}

const MLAgentPanel: React.FC<MLAgentPanelProps> = ({ userId }) => {
  const [activeTab, setActiveTab] = useState('health');
  const [loading, setLoading] = useState(false);
  const [healthAgentResults, setHealthAgentResults] = useState<any>(null);
  const [safetyAgentResults, setSafetyAgentResults] = useState<any>(null);
  const [reminderAgentResults, setReminderAgentResults] = useState<any>(null);
  
  const handleTriggerHealthAgent = async () => {
    if (!userId) {
      toast.error('Please select a user first');
      return;
    }
    
    setLoading(true);
    setHealthAgentResults(null);
    try {
      const response = await triggerHealthAgent(userId);
      if (response.success) {
        toast.success('Health agent triggered successfully');
        setHealthAgentResults(response.results || { issues: [] });
      } else {
        toast.error(`Failed to trigger health agent: ${response.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error triggering health agent:', error);
      toast.error('An error occurred while triggering health agent');
    } finally {
      setLoading(false);
    }
  };
  
  const handleTriggerSafetyAgent = async () => {
    if (!userId) {
      toast.error('Please select a user first');
      return;
    }
    
    setLoading(true);
    setSafetyAgentResults(null);
    try {
      const response = await triggerSafetyAgent(userId);
      if (response.success) {
        toast.success('Safety agent triggered successfully');
        setSafetyAgentResults(response.results || { issues: [] });
      } else {
        toast.error(`Failed to trigger safety agent: ${response.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error triggering safety agent:', error);
      toast.error('An error occurred while triggering safety agent');
    } finally {
      setLoading(false);
    }
  };
  
  const handleTriggerReminderAgent = async () => {
    if (!userId) {
      toast.error('Please select a user first');
      return;
    }
    
    setLoading(true);
    setReminderAgentResults(null);
    try {
      const response = await triggerReminderAgent(userId);
      if (response.success) {
        toast.success('Reminder agent triggered successfully');
        setReminderAgentResults(response.results || { issues: [] });
      } else {
        toast.error(`Failed to trigger reminder agent: ${response.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error triggering reminder agent:', error);
      toast.error('An error occurred while triggering reminder agent');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <Brain className="mr-2 h-5 w-5" />
          Agent Control Panel
        </CardTitle>
        <CardDescription>
          Trigger ML-powered agents to analyze user data and provide recommendations
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="health" className="flex items-center">
              <Activity className="mr-2 h-4 w-4" />
              Health
            </TabsTrigger>
            <TabsTrigger value="safety" className="flex items-center">
              <Shield className="mr-2 h-4 w-4" />
              Safety
            </TabsTrigger>
            <TabsTrigger value="reminders" className="flex items-center">
              <Bell className="mr-2 h-4 w-4" />
              Reminders
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="health" className="space-y-4 pt-4">
            {healthAgentResults && (
              <AgentResultDisplay
                title="Health Agent Analysis"
                results={healthAgentResults}
                success={true}
              />
            )}
            
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Analyze health data to detect anomalies and predict health issues
              </p>
              
              <div className="bg-muted p-3 rounded-md">
                <h4 className="font-medium mb-2">ML Capabilities</h4>
                <ul className="text-sm space-y-1 list-disc list-inside">
                  <li>Anomaly detection using isolation forest</li>
                  <li>Supervised classification of health issues</li>
                  <li>Severity prediction for health anomalies</li>
                </ul>
              </div>
              
              <Button 
                onClick={handleTriggerHealthAgent}
                disabled={loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Trigger Health Agent'
                )}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="safety" className="space-y-4 pt-4">
            {safetyAgentResults && (
              <AgentResultDisplay
                title="Safety Agent Analysis"
                results={safetyAgentResults}
                success={true}
              />
            )}
            
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Analyze safety data to identify risks and detect potential falls or emergencies
              </p>
              
              <div className="bg-muted p-3 rounded-md">
                <h4 className="font-medium mb-2">ML Capabilities</h4>
                <ul className="text-sm space-y-1 list-disc list-inside">
                  <li>Fall detection with pattern recognition</li>
                  <li>Time-series analysis of activity levels</li>
                  <li>Risk assessment based on multiple factors</li>
                </ul>
              </div>
              
              <Button 
                onClick={handleTriggerSafetyAgent}
                disabled={loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Trigger Safety Agent'
                )}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="reminders" className="space-y-4 pt-4">
            {reminderAgentResults && (
              <AgentResultDisplay
                title="Reminder Agent Analysis"
                results={reminderAgentResults}
                success={true}
              />
            )}
            
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Optimize and personalize reminders based on user's history and preferences
              </p>
              
              <div className="bg-muted p-3 rounded-md">
                <h4 className="font-medium mb-2">ML Capabilities</h4>
                <ul className="text-sm space-y-1 list-disc list-inside">
                  <li>Reminder effectiveness analysis</li>
                  <li>Optimal timing prediction</li>
                  <li>Content personalization</li>
                </ul>
              </div>
              
              <Button 
                onClick={handleTriggerReminderAgent}
                disabled={loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Trigger Reminder Agent'
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default MLAgentPanel; 