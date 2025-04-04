import React, { useState, useEffect } from 'react';
import { Box, Button, IconButton, TextField, Menu, MenuItem, Select, FormControl, InputLabel, Typography, Snackbar, Alert } from '@mui/material';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import SettingsVoiceIcon from '@mui/icons-material/SettingsVoice';
import { backendApi } from '../services/backendApi';

interface Voice {
  id: string;
  name: string;
  gender: string;
  engine: string;
}

interface VoiceReminderProps {
  reminderId?: number;
  userId?: number;
  defaultMessage?: string;
  compact?: boolean; // If true, just shows an icon button
}

const VoiceReminder: React.FC<VoiceReminderProps> = ({ 
  reminderId, 
  userId, 
  defaultMessage = "No message provided",
  compact = false
}) => {
  const [message, setMessage] = useState<string>(defaultMessage);
  const [priority, setPriority] = useState<string>('medium');
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [success, setSuccess] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  // Fetch available voices on component mount
  useEffect(() => {
    const fetchVoices = async () => {
      try {
        const response = await backendApi.get('/api/voices');
        if (response.status === 200 && response.data.voices) {
          setVoices(response.data.voices);
          // Set default voice if available
          if (response.data.voices.length > 0) {
            setSelectedVoice(response.data.voices[0].id);
          }
        }
      } catch (error) {
        console.error('Error fetching voices:', error);
      }
    };

    fetchVoices();
  }, []);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    if (compact) {
      // In compact mode, just speak the reminder directly
      speakReminder();
    } else {
      // In full mode, open the menu
      setAnchorEl(event.currentTarget);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const speakReminder = async () => {
    if (reminderId) {
      try {
        setLoading(true);
        const response = await backendApi.post(`/api/reminders/speak/${reminderId}`);
        if (response.status === 200) {
          setSuccess(true);
        } else {
          setError('Failed to speak reminder');
        }
      } catch (error) {
        console.error('Error speaking reminder:', error);
        setError('Error speaking reminder');
      } finally {
        setLoading(false);
        handleClose();
      }
    } else {
      speakCustomMessage();
    }
  };

  const speakCustomMessage = async () => {
    try {
      setLoading(true);
      
      // Prepare voice settings
      const voiceSettings: any = {};
      if (selectedVoice) {
        voiceSettings.voice = selectedVoice;
      }
      
      const response = await backendApi.post('/api/speak', {
        message,
        user_id: userId,
        priority,
        voice_settings: voiceSettings
      });
      
      if (response.status === 200) {
        setSuccess(true);
      } else {
        setError('Failed to speak message');
      }
    } catch (error) {
      console.error('Error speaking custom message:', error);
      setError('Error speaking custom message');
    } finally {
      setLoading(false);
      handleClose();
    }
  };

  const handleSnackbarClose = () => {
    setSuccess(false);
    setError(null);
  };

  // Compact version - just an icon button
  if (compact) {
    return (
      <>
        <IconButton 
          aria-label="speak reminder" 
          onClick={handleClick}
          disabled={loading}
          color="primary"
          size="small"
        >
          <VolumeUpIcon />
        </IconButton>
        
        <Snackbar 
          open={success} 
          autoHideDuration={3000} 
          onClose={handleSnackbarClose}
        >
          <Alert severity="success">Voice reminder sent!</Alert>
        </Snackbar>
        
        <Snackbar 
          open={!!error} 
          autoHideDuration={3000} 
          onClose={handleSnackbarClose}
        >
          <Alert severity="error">{error}</Alert>
        </Snackbar>
      </>
    );
  }

  // Full version with dialog
  return (
    <>
      <Button
        variant="outlined"
        startIcon={<SettingsVoiceIcon />}
        onClick={handleClick}
        disabled={loading}
      >
        Voice Reminder
      </Button>
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        PaperProps={{
          style: {
            width: '350px',
            padding: '16px',
          },
        }}
      >
        <Typography variant="h6" gutterBottom>
          Voice Reminder
        </Typography>
        
        <TextField
          label="Message"
          fullWidth
          multiline
          rows={3}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          margin="normal"
          variant="outlined"
        />
        
        <Box sx={{ mt: 2, mb: 2 }}>
          <FormControl fullWidth variant="outlined">
            <InputLabel>Priority</InputLabel>
            <Select
              value={priority}
              onChange={(e) => setPriority(e.target.value as string)}
              label="Priority"
            >
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>
        </Box>
        
        {voices.length > 0 && (
          <Box sx={{ mt: 2, mb: 2 }}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Voice</InputLabel>
              <Select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value as string)}
                label="Voice"
              >
                {voices.map((voice) => (
                  <MenuItem key={voice.id} value={voice.id}>
                    {voice.name} ({voice.gender})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        )}
        
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button 
            onClick={handleClose} 
            sx={{ mr: 1 }}
          >
            Cancel
          </Button>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={reminderId ? speakReminder : speakCustomMessage}
            disabled={loading || !message}
          >
            Speak Now
          </Button>
        </Box>
      </Menu>
      
      <Snackbar 
        open={success} 
        autoHideDuration={3000} 
        onClose={handleSnackbarClose}
      >
        <Alert severity="success">Voice reminder sent!</Alert>
      </Snackbar>
      
      <Snackbar 
        open={!!error} 
        autoHideDuration={3000} 
        onClose={handleSnackbarClose}
      >
        <Alert severity="error">{error}</Alert>
      </Snackbar>
    </>
  );
};

export default VoiceReminder; 