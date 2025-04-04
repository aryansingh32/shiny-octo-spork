
import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { useToast } from "@/hooks/use-toast";

interface VoiceContextType {
  voiceEnabled: boolean;
  toggleVoice: () => void;
  isListening: boolean;
  startListening: () => void;
  stopListening: () => void;
  speak: (text: string) => void;
}

const VoiceContext = createContext<VoiceContextType | undefined>(undefined);

export function VoiceProvider({ children }: { children: React.ReactNode }) {
  const [voiceEnabled, setVoiceEnabled] = useState(() => {
    return localStorage.getItem("voiceEnabled") === "true";
  });
  const [isListening, setIsListening] = useState(false);
  const { toast } = useToast();

  // Mock Speech Recognition API for now
  // In a real implementation, you'd use the Web Speech API or a third-party library
  const startListening = useCallback(() => {
    if (!voiceEnabled) return;
    
    setIsListening(true);
    toast({
      title: "Voice Recognition Active",
      description: "I'm listening for commands...",
      duration: 3000,
    });
    
    // Simulate stopping after a few seconds
    setTimeout(() => {
      stopListening();
    }, 5000);
  }, [voiceEnabled, toast]);

  const stopListening = useCallback(() => {
    setIsListening(false);
  }, []);

  // Speech synthesis for reading text
  const speak = useCallback((text: string) => {
    if (!voiceEnabled) return;
    
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9; // Slightly slower for better comprehension
      utterance.pitch = 1;
      
      // Get available voices and try to find a clear one
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(voice => 
        voice.name.includes('Female') || 
        voice.name.includes('Samantha') ||
        voice.name.includes('Karen')
      );
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }
      
      window.speechSynthesis.speak(utterance);
    }
  }, [voiceEnabled]);

  const toggleVoice = () => {
    setVoiceEnabled(prev => {
      const newValue = !prev;
      localStorage.setItem("voiceEnabled", newValue.toString());
      
      if (newValue) {
        toast({
          title: "Voice Features Enabled",
          description: "You can now use voice commands and text-to-speech",
        });
        
        // Welcome message
        setTimeout(() => {
          speak("Voice features are now enabled. I can read content and listen for commands.");
        }, 500);
      } else {
        toast({
          title: "Voice Features Disabled",
          description: "Voice commands and text-to-speech are turned off",
        });
      }
      
      return newValue;
    });
  };

  // Initialize speech synthesis voices when available
  useEffect(() => {
    if ('speechSynthesis' in window) {
      // Chrome needs this to get the voices
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }
  }, []);

  return (
    <VoiceContext.Provider value={{ 
      voiceEnabled, 
      toggleVoice, 
      isListening,
      startListening,
      stopListening,
      speak
    }}>
      {children}
    </VoiceContext.Provider>
  );
}

export function useVoice() {
  const context = useContext(VoiceContext);
  if (context === undefined) {
    throw new Error("useVoice must be used within a VoiceProvider");
  }
  return context;
}
