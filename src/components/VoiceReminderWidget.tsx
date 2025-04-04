import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Loader2, Volume2 } from 'lucide-react';
import { toast } from 'sonner';
import { speakCustomMessage } from '@/services/backendApi';

interface VoiceReminderWidgetProps {
  userId?: string;
}

const VoiceReminderWidget: React.FC<VoiceReminderWidgetProps> = ({ userId }) => {
  const [message, setMessage] = useState<string>('');
  const [priority, setPriority] = useState<string>('medium');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  
  const handleSendVoiceReminder = async () => {
    if (!message.trim()) {
      toast.error('Please enter a message');
      return;
    }
    
    setIsSubmitting(true);
    try {
      await speakCustomMessage(
        message,
        userId ? parseInt(userId) : undefined,
        priority
      );
      toast.success('Voice reminder sent successfully');
      setMessage('');
    } catch (error) {
      console.error('Error sending voice reminder:', error);
      toast.error('Failed to send voice reminder');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-md font-medium flex items-center">
          <Volume2 className="mr-2 h-4 w-4" />
          Voice Reminder
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="voice-message">Message</Label>
            <Textarea
              id="voice-message"
              placeholder="Enter voice message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="min-h-[80px]"
            />
          </div>
          
          <div className="space-y-1">
            <Label htmlFor="priority">Priority</Label>
            <Select 
              value={priority} 
              onValueChange={setPriority}
            >
              <SelectTrigger id="priority">
                <SelectValue placeholder="Select priority" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="high">High</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <Button 
            onClick={handleSendVoiceReminder} 
            disabled={isSubmitting || !message.trim()} 
            className="w-full"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Volume2 className="mr-2 h-4 w-4" />
                Send Voice Reminder
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default VoiceReminderWidget; 