import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle, 
  CardDescription 
} from '@/components/ui/card';
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from '@/components/ui/tabs';
import { 
  AlertCircle, 
  Activity, 
  Shield, 
  Bell, 
  Brain, 
  TrendingUp
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { getMLInsights, getMLModelStatus, mlFetch } from '@/services/backendApi';

interface MLInsightsPanelProps {
  userId?: string;
}

interface MLInsight {
  id: string;
  type: string;
  title: string;
  description: string;
  confidence: number;
  timestamp: string;
  source: string;
  severity?: 'low' | 'medium' | 'high';
  actions?: Array<{
    id: string;
    label: string;
    endpoint: string;
    method: 'GET' | 'POST' | 'PUT';
    payload?: any;
  }>;
}

const MLInsightsPanel: React.FC<MLInsightsPanelProps> = ({ userId }) => {
  const [activeTab, setActiveTab] = useState('health');
  const [insights, setInsights] = useState<MLInsight[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [modelStatus, setModelStatus] = useState<Record<string, boolean>>({
    'health': false,
    'safety': false,
    'reminder': false
  });

  // Fetch insights from the ML agents
  useEffect(() => {
    if (!userId) return;
    
    const fetchInsights = async () => {
      setLoading(true);
      try {
        // This endpoint would aggregate insights from all agents for this user
        const response = await getMLInsights(userId);
        if (response.insights) {
          setInsights(response.insights);
        }
        
        // Check model status
        const statusResponse = await getMLModelStatus();
        if (statusResponse.models) {
          setModelStatus(statusResponse.models);
        }
      } catch (error) {
        console.error('Error fetching ML insights:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchInsights();
    // Set up polling for real-time updates
    const interval = setInterval(fetchInsights, 30000); // every 30 seconds
    
    return () => clearInterval(interval);
  }, [userId]);

  const handleAction = async (action: any) => {
    try {
      await mlFetch(action.endpoint, {
        method: action.method,
        ...(action.payload && { body: JSON.stringify(action.payload) })
      });
      // Refresh insights after action
      const response = await getMLInsights(userId);
      if (response.insights) {
        setInsights(response.insights);
      }
    } catch (error) {
      console.error(`Error performing action ${action.label}:`, error);
    }
  };

  const filteredInsights = insights.filter(insight => 
    activeTab === 'all' || insight.source.toLowerCase().includes(activeTab)
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              <Brain className="mr-2 h-5 w-5 text-primary" />
              ML Insights & Agent Interaction
            </CardTitle>
            <CardDescription>
              Machine learning powered insights and recommendations
            </CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            {Object.entries(modelStatus).map(([key, status]) => (
              <Badge 
                key={key}
                variant={status ? "default" : "outline"}
                className={status ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-800"}
              >
                {key.charAt(0).toUpperCase() + key.slice(1)} Models: {status ? 'Active' : 'Fallback'}
              </Badge>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="health" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-4">
            <TabsTrigger value="health" className="flex items-center">
              <Activity className="mr-2 h-4 w-4" />
              Health
            </TabsTrigger>
            <TabsTrigger value="safety" className="flex items-center">
              <Shield className="mr-2 h-4 w-4" />
              Safety
            </TabsTrigger>
            <TabsTrigger value="reminder" className="flex items-center">
              <Bell className="mr-2 h-4 w-4" />
              Reminders
            </TabsTrigger>
            <TabsTrigger value="all" className="flex items-center">
              <TrendingUp className="mr-2 h-4 w-4" />
              All Insights
            </TabsTrigger>
          </TabsList>
          
          <div className="mt-4">
            {loading ? (
              <div className="text-center py-8">
                <p>Loading insights...</p>
              </div>
            ) : filteredInsights.length > 0 ? (
              <div className="space-y-4">
                {filteredInsights.map((insight) => (
                  <Card key={insight.id} className="overflow-hidden">
                    <div className={`h-1 ${
                      insight.severity === 'high' ? 'bg-red-500' : 
                      insight.severity === 'medium' ? 'bg-yellow-500' : 
                      'bg-blue-500'
                    }`} />
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <p className="font-medium">{insight.title}</p>
                          <p className="text-sm text-muted-foreground">{insight.description}</p>
                        </div>
                        <Badge variant="outline" className="ml-2">
                          {insight.source}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center mt-2 text-xs text-muted-foreground">
                        <span>Confidence: {(insight.confidence * 100).toFixed(1)}%</span>
                        <span className="mx-2">•</span>
                        <span>{new Date(insight.timestamp).toLocaleString()}</span>
                      </div>
                      
                      {insight.actions && insight.actions.length > 0 && (
                        <>
                          <Separator className="my-3" />
                          <div className="flex flex-wrap gap-2">
                            {insight.actions.map(action => (
                              <Button 
                                key={action.id}
                                variant="outline" 
                                size="sm"
                                onClick={() => handleAction(action)}
                              >
                                {action.label}
                              </Button>
                            ))}
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>No ML insights available for {activeTab === 'all' ? 'any agent' : activeTab}</p>
                <p className="text-sm mt-1">Insights will appear as the agents analyze data</p>
              </div>
            )}
          </div>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default MLInsightsPanel; 