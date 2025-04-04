import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { ArrowUpFromLine, Activity, Shield, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { submitHealthData, submitSafetyData } from '@/services/backendApi';
import AgentResultDisplay from './AgentResultDisplay';

interface DataSubmissionFormProps {
  userId?: string;
}

const DataSubmissionForm: React.FC<DataSubmissionFormProps> = ({ userId }) => {
  const [activeTab, setActiveTab] = useState('health');
  const [loading, setLoading] = useState(false);
  const [healthResults, setHealthResults] = useState<any>(null);
  const [safetyResults, setSafetyResults] = useState<any>(null);
  
  // Health data form
  const [healthData, setHealthData] = useState({
    heart_rate: '',
    blood_pressure_systolic: '',
    blood_pressure_diastolic: '',
    temperature: '',
    oxygen_saturation: '',
    glucose_level: '',
    weight: '',
  });
  
  // Safety data form
  const [safetyData, setHealthSafetyData] = useState({
    location: 'bedroom',
    movement_type: 'walking',
    activity_level: 'moderate',
    fall_detected: 'no',
    time_inactive: '0',
  });

  const handleHealthDataChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setHealthData(prev => ({ ...prev, [name]: value }));
  };
  
  const handleSafetyInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setHealthSafetyData(prev => ({ ...prev, [name]: value }));
  };
  
  const handleSafetySelectChange = (name: string, value: string) => {
    setHealthSafetyData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmitHealthData = async () => {
    if (!userId) {
      toast.error('Please select a user first');
      return;
    }
    
    setLoading(true);
    setHealthResults(null);
    try {
      // Convert string values to numbers
      const formattedData = Object.entries(healthData).reduce((acc, [key, value]) => {
        return { ...acc, [key]: value === '' ? null : parseFloat(value) };
      }, {});
      
      const response = await submitHealthData(userId, {
        ...formattedData,
        timestamp: new Date().toISOString()
      });
      
      if (response.success) {
        toast.success('Health data submitted successfully');
        // Store the results for display
        setHealthResults(response.results || { issues: [] });
        // Clear form
        setHealthData({
          heart_rate: '',
          blood_pressure_systolic: '',
          blood_pressure_diastolic: '',
          temperature: '',
          oxygen_saturation: '',
          glucose_level: '',
          weight: '',
        });
      } else {
        toast.error('Failed to submit health data');
      }
    } catch (error) {
      console.error('Error submitting health data:', error);
      toast.error('An error occurred while submitting health data');
    } finally {
      setLoading(false);
    }
  };
  
  const handleSubmitSafetyData = async () => {
    if (!userId) {
      toast.error('Please select a user first');
      return;
    }
    
    setLoading(true);
    setSafetyResults(null);
    try {
      const fallDetected = safetyData.fall_detected === 'yes';
      const timeInactive = parseInt(safetyData.time_inactive);
      
      const response = await submitSafetyData(userId, {
        location: safetyData.location,
        movement_type: safetyData.movement_type,
        activity_level: safetyData.activity_level,
        fall_detected: fallDetected,
        time_inactive: isNaN(timeInactive) ? 0 : timeInactive,
        timestamp: new Date().toISOString()
      });
      
      if (response.success) {
        toast.success('Safety data submitted successfully');
        // Store the results for display
        setSafetyResults(response.results || { issues: [] });
        // Reset fall detection and time inactive
        setHealthSafetyData(prev => ({
          ...prev,
          fall_detected: 'no',
          time_inactive: '0'
        }));
      } else {
        toast.error('Failed to submit safety data');
      }
    } catch (error) {
      console.error('Error submitting safety data:', error);
      toast.error('An error occurred while submitting safety data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center">
          <ArrowUpFromLine className="mr-2 h-5 w-5" />
          Submit Data for ML Analysis
        </CardTitle>
        <CardDescription>
          Submit health and safety data to be analyzed by the machine learning agents
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="health" className="flex items-center">
              <Activity className="mr-2 h-4 w-4" />
              Health Data
            </TabsTrigger>
            <TabsTrigger value="safety" className="flex items-center">
              <Shield className="mr-2 h-4 w-4" />
              Safety Data
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="health" className="space-y-4 pt-4">
            {healthResults && (
              <AgentResultDisplay
                title="Health Data Analysis"
                results={healthResults}
                success={true}
              />
            )}
          
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="heart_rate">Heart Rate (bpm)</Label>
                <Input
                  id="heart_rate"
                  name="heart_rate"
                  type="number"
                  placeholder="e.g., 75"
                  value={healthData.heart_rate}
                  onChange={handleHealthDataChange}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="temperature">Temperature (°C)</Label>
                <Input
                  id="temperature"
                  name="temperature"
                  type="number"
                  step="0.1"
                  placeholder="e.g., 36.8"
                  value={healthData.temperature}
                  onChange={handleHealthDataChange}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="blood_pressure_systolic">Systolic BP (mmHg)</Label>
                <Input
                  id="blood_pressure_systolic"
                  name="blood_pressure_systolic"
                  type="number"
                  placeholder="e.g., 120"
                  value={healthData.blood_pressure_systolic}
                  onChange={handleHealthDataChange}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="blood_pressure_diastolic">Diastolic BP (mmHg)</Label>
                <Input
                  id="blood_pressure_diastolic"
                  name="blood_pressure_diastolic"
                  type="number"
                  placeholder="e.g., 80"
                  value={healthData.blood_pressure_diastolic}
                  onChange={handleHealthDataChange}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="oxygen_saturation">Oxygen Saturation (%)</Label>
                <Input
                  id="oxygen_saturation"
                  name="oxygen_saturation"
                  type="number"
                  placeholder="e.g., 98"
                  value={healthData.oxygen_saturation}
                  onChange={handleHealthDataChange}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="glucose_level">Glucose Level (mg/dL)</Label>
                <Input
                  id="glucose_level"
                  name="glucose_level"
                  type="number"
                  placeholder="e.g., 100"
                  value={healthData.glucose_level}
                  onChange={handleHealthDataChange}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="weight">Weight (kg)</Label>
                <Input
                  id="weight"
                  name="weight"
                  type="number"
                  step="0.1"
                  placeholder="e.g., 70.5"
                  value={healthData.weight}
                  onChange={handleHealthDataChange}
                />
              </div>
            </div>
            
            <Button 
              className="w-full mt-4" 
              onClick={handleSubmitHealthData}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                'Submit Health Data for Analysis'
              )}
            </Button>
          </TabsContent>
          
          <TabsContent value="safety" className="space-y-4 pt-4">
            {safetyResults && (
              <AgentResultDisplay
                title="Safety Data Analysis"
                results={safetyResults}
                success={true}
              />
            )}
            
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="location">Current Location</Label>
                <Select
                  value={safetyData.location}
                  onValueChange={(value) => handleSafetySelectChange('location', value)}
                >
                  <SelectTrigger id="location">
                    <SelectValue placeholder="Select location" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bedroom">Bedroom</SelectItem>
                    <SelectItem value="bathroom">Bathroom</SelectItem>
                    <SelectItem value="kitchen">Kitchen</SelectItem>
                    <SelectItem value="living_room">Living Room</SelectItem>
                    <SelectItem value="outside">Outside</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="movement_type">Movement Type</Label>
                <Select
                  value={safetyData.movement_type}
                  onValueChange={(value) => handleSafetySelectChange('movement_type', value)}
                >
                  <SelectTrigger id="movement_type">
                    <SelectValue placeholder="Select movement type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="walking">Walking</SelectItem>
                    <SelectItem value="sitting">Sitting</SelectItem>
                    <SelectItem value="standing">Standing</SelectItem>
                    <SelectItem value="lying_down">Lying Down</SelectItem>
                    <SelectItem value="unknown">Unknown</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="activity_level">Activity Level</Label>
                <Select
                  value={safetyData.activity_level}
                  onValueChange={(value) => handleSafetySelectChange('activity_level', value)}
                >
                  <SelectTrigger id="activity_level">
                    <SelectValue placeholder="Select activity level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="moderate">Moderate</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="fall_detected">Fall Detected</Label>
                <Select
                  value={safetyData.fall_detected}
                  onValueChange={(value) => handleSafetySelectChange('fall_detected', value)}
                >
                  <SelectTrigger id="fall_detected">
                    <SelectValue placeholder="Fall detected?" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="no">No</SelectItem>
                    <SelectItem value="yes">Yes</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="time_inactive">Time Inactive (minutes)</Label>
                <Input
                  id="time_inactive"
                  name="time_inactive"
                  type="number"
                  min="0"
                  placeholder="e.g., 0"
                  value={safetyData.time_inactive}
                  onChange={handleSafetyInputChange}
                />
              </div>
            </div>
            
            <Button 
              className="w-full mt-4" 
              onClick={handleSubmitSafetyData}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                'Submit Safety Data for Analysis'
              )}
            </Button>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default DataSubmissionForm; 