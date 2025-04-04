import os
import sys
import logging
import json
import tempfile
from datetime import datetime
import subprocess
import threading
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for cloud environment detection
IS_CLOUD_ENV = os.environ.get('RENDER', False) or os.environ.get('CLOUD_ENV', False)
if IS_CLOUD_ENV:
    logger.info("Detected cloud environment, will use web-based TTS only")

class VoiceReminderService:
    """
    A cross-platform voice reminder service that supports multiple text-to-speech engines.
    This service can fall back to different engines based on platform availability and 
    supports saving audio files for web playback.
    """
    
    def __init__(self, config=None):
        """
        Initialize the voice reminder service with the given configuration.
        
        Args:
            config (dict, optional): Configuration parameters for the service.
        """
        self.config = config or {}
        self.voice_queue = queue.Queue()
        self.audio_dir = self.config.get('audio_dir', os.path.join(os.path.dirname(__file__), 'audio_files'))
        self.tts_engine = None
        self.is_processing = False
        self.processing_thread = None
        
        # Create audio directory if it doesn't exist
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Default voice settings
        self.default_settings = {
            'rate': self.config.get('rate', 150),  # Speed of speech
            'volume': self.config.get('volume', 1.0),  # Volume level (0.0 to 1.0)
            'voice': self.config.get('voice', None),  # Default voice
            'save_audio': self.config.get('save_audio', True),  # Save audio files for web playback
            'play_audio': not IS_CLOUD_ENV and self.config.get('play_audio', True),  # Play audio immediately (not in cloud)
            'language': self.config.get('language', 'en-US'),  # Default language
            'gender': self.config.get('gender', None)  # Voice gender preference
        }
        
        # Platform detection
        self.platform = self._detect_platform()
        logger.info(f"Detected platform: {self.platform}")
        
        # Available engines
        self.available_engines = self._discover_available_engines()
        logger.info(f"Available TTS engines: {', '.join(self.available_engines)}")
        
        # Force web engine in cloud environments
        if IS_CLOUD_ENV:
            logger.info("Using web-based TTS for cloud environment")
            self.tts_engine = 'web'
        else:
            # Initialize the best available engine
            self._initialize_tts_engine()
        
        # Start processing thread
        self._start_processing_thread()
        
    def _detect_platform(self):
        """Detect the current platform."""
        if IS_CLOUD_ENV:
            return 'cloud'
        elif sys.platform.startswith('win'):
            return 'windows'
        elif sys.platform.startswith('linux'):
            return 'linux'
        elif sys.platform.startswith('darwin'):
            return 'macos'
        else:
            return 'unknown'
            
    def _discover_available_engines(self):
        """Discover available TTS engines on the current platform."""
        available_engines = []
        
        # Check for pyttsx3
        try:
            import pyttsx3
            available_engines.append('pyttsx3')
        except ImportError:
            pass
            
        # Check for gTTS (Google Text-to-Speech)
        try:
            from gtts import gTTS
            available_engines.append('gtts')
        except ImportError:
            pass
            
        # Check for espeak on Linux
        if self.platform == 'linux':
            try:
                result = subprocess.run(['which', 'espeak'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
                if result.returncode == 0:
                    available_engines.append('espeak')
            except:
                pass
        
        # Check for Microsoft SAPI on Windows
        if self.platform == 'windows':
            try:
                import comtypes.client
                available_engines.append('sapi')
            except ImportError:
                pass
                
        # Check for festival on Linux
        if self.platform == 'linux':
            try:
                result = subprocess.run(['which', 'festival'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
                if result.returncode == 0:
                    available_engines.append('festival')
            except:
                pass
                
        # Always add 'web' as a fallback option that saves the text for web-based playback
        available_engines.append('web')
        
        return available_engines
        
    def _initialize_tts_engine(self):
        """Initialize the best available TTS engine."""
        # Order of preference based on quality and feature set
        preferred_order = ['pyttsx3', 'sapi', 'gtts', 'espeak', 'festival', 'web']
        
        # User specified engine in config
        user_engine = self.config.get('engine')
        if user_engine and user_engine in self.available_engines:
            preferred_order.insert(0, user_engine)
        
        # Find the first available engine in order of preference
        for engine in preferred_order:
            if engine in self.available_engines:
                self.tts_engine = engine
                logger.info(f"Selected TTS engine: {engine}")
                
                # Initialize specific engines if needed
                if engine == 'pyttsx3':
                    try:
                        import pyttsx3
                        self._pyttsx3_engine = pyttsx3.init()
                        self._pyttsx3_engine.setProperty('rate', self.default_settings['rate'])
                        self._pyttsx3_engine.setProperty('volume', self.default_settings['volume'])
                        
                        # Set voice if specified
                        if self.default_settings['voice'] or self.default_settings['gender']:
                            voices = self._pyttsx3_engine.getProperty('voices')
                            selected_voice = None
                            
                            # Filter by specified voice id or gender
                            for voice in voices:
                                # Direct voice ID match
                                if self.default_settings['voice'] and voice.id == self.default_settings['voice']:
                                    selected_voice = voice.id
                                    break
                                # Gender match (if voice id not specified)
                                elif self.default_settings['gender'] and not self.default_settings['voice']:
                                    if (self.default_settings['gender'].lower() == 'male' and 'male' in voice.name.lower()) or \
                                       (self.default_settings['gender'].lower() == 'female' and 'female' in voice.name.lower()):
                                        selected_voice = voice.id
                                        break
                            
                            if selected_voice:
                                self._pyttsx3_engine.setProperty('voice', selected_voice)
                    except Exception as e:
                        logger.error(f"Failed to initialize pyttsx3: {e}")
                        continue
                
                break
        
        if not self.tts_engine:
            logger.warning("No TTS engine available, falling back to text-only mode")
            self.tts_engine = 'text'
    
    def _start_processing_thread(self):
        """Start the thread that processes voice reminders from the queue."""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_voice_queue, daemon=True)
        self.processing_thread.start()
        logger.info("Started voice reminder processing thread")
    
    def _process_voice_queue(self):
        """Process voice reminders from the queue."""
        while self.is_processing:
            try:
                # Get reminder from queue with timeout (allows thread to check is_processing)
                reminder = self.voice_queue.get(timeout=1)
                logger.info(f"Processing voice reminder: {reminder['text'][:50]}...")
                
                # Process the reminder
                self._speak_reminder(reminder)
                
                # Mark task as done
                self.voice_queue.task_done()
            except queue.Empty:
                # No reminders in queue, continue waiting
                pass
            except Exception as e:
                logger.error(f"Error processing voice reminder: {e}")
    
    def _speak_reminder(self, reminder):
        """Speak the reminder using the selected TTS engine."""
        text = reminder['text']
        settings = {**self.default_settings, **reminder.get('settings', {})}
        user_id = reminder.get('user_id')
        reminder_id = reminder.get('reminder_id')
        
        try:
            # File path for saving audio (if needed)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            file_name = f"reminder_{user_id}_{reminder_id}_{timestamp}"
            audio_path = os.path.join(self.audio_dir, file_name)
            
            # Process based on selected engine
            if self.tts_engine == 'pyttsx3' and hasattr(self, '_pyttsx3_engine'):
                # Update properties if they were changed
                self._pyttsx3_engine.setProperty('rate', settings['rate'])
                self._pyttsx3_engine.setProperty('volume', settings['volume'])
                
                if settings['save_audio']:
                    # Save to file
                    self._pyttsx3_engine.save_to_file(text, f"{audio_path}.mp3")
                    self._pyttsx3_engine.runAndWait()
                    logger.info(f"Saved audio to {audio_path}.mp3")
                
                if settings['play_audio']:
                    # Play audio
                    self._pyttsx3_engine.say(text)
                    self._pyttsx3_engine.runAndWait()
                
            elif self.tts_engine == 'gtts':
                from gtts import gTTS
                
                # Create gTTS object
                tts = gTTS(text=text, lang=settings['language'], slow=False)
                
                # Save audio to file
                tts.save(f"{audio_path}.mp3")
                logger.info(f"Saved audio to {audio_path}.mp3")
                
                if settings['play_audio']:
                    # Play audio using platform-specific method
                    self._play_audio_file(f"{audio_path}.mp3")
            
            elif self.tts_engine == 'espeak':
                # Save to file first (if requested)
                if settings['save_audio']:
                    subprocess.run([
                        'espeak', '-w', f"{audio_path}.wav", 
                        '-s', str(settings['rate']), 
                        '-a', str(int(settings['volume'] * 100)),
                        text
                    ])
                    logger.info(f"Saved audio to {audio_path}.wav")
                
                # Play directly if requested
                if settings['play_audio']:
                    subprocess.run([
                        'espeak',
                        '-s', str(settings['rate']), 
                        '-a', str(int(settings['volume'] * 100)),
                        text
                    ])
            
            elif self.tts_engine == 'festival':
                # Create temporary file with text
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(text)
                    temp_file = f.name
                
                try:
                    # Save to file if requested
                    if settings['save_audio']:
                        subprocess.run([
                            'text2wave', temp_file, 
                            '-o', f"{audio_path}.wav"
                        ])
                        logger.info(f"Saved audio to {audio_path}.wav")
                    
                    # Play directly if requested
                    if settings['play_audio']:
                        # Festival can pipe directly to audio output
                        subprocess.run(f'echo "{text}" | festival --tts', shell=True)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file)
            
            elif self.tts_engine == 'sapi':
                import comtypes.client
                
                # Initialize SAPI
                speaker = comtypes.client.CreateObject("SAPI.SpVoice")
                
                # Set rate and volume
                # SAPI rate range is -10 to 10, where 0 is normal
                # Convert our 50-250 to -10 to 10
                rate_normalized = ((settings['rate'] - 150) / 100) * 10
                speaker.Rate = rate_normalized
                
                # SAPI volume range is 0 to 100
                speaker.Volume = int(settings['volume'] * 100)
                
                # Set voice if specified
                if settings['voice'] or settings['gender']:
                    voices = speaker.GetVoices()
                    for i in range(voices.Count):
                        voice = voices.Item(i)
                        # Direct voice ID match
                        if settings['voice'] and settings['voice'] in str(voice.Id):
                            speaker.Voice = voice
                            break
                        # Gender match
                        elif settings['gender'] and not settings['voice']:
                            gender = 'male' if 'male' in settings['gender'].lower() else 'female'
                            desc = voice.GetDescription()
                            if gender in desc.lower():
                                speaker.Voice = voice
                                break
                
                # Save to file if requested
                if settings['save_audio']:
                    stream = comtypes.client.CreateObject("SAPI.SpFileStream")
                    stream.Open(f"{audio_path}.wav", 3)  # 3 = SSFMCreateForWrite
                    speaker.AudioOutputStream = stream
                    speaker.Speak(text)
                    stream.Close()
                    logger.info(f"Saved audio to {audio_path}.wav")
                
                # Play directly if requested
                if settings['play_audio']:
                    # Reset audio output to default
                    speaker.AudioOutputStream = None
                    speaker.Speak(text)
            
            elif self.tts_engine == 'web':
                # For web deployment, just save the text to a JSON file
                with open(f"{audio_path}.json", 'w') as f:
                    json.dump({
                        'text': text,
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'reminder_id': reminder_id,
                        'settings': settings
                    }, f)
                logger.info(f"Saved reminder text to {audio_path}.json for web playback")
                
                # You can also save a placeholder audio file or empty MP3
                # This is useful for web interfaces that expect an audio file
                self._create_placeholder_audio(f"{audio_path}.mp3")
                
            # Update reminder status in the database (if callback provided)
            if 'callback' in reminder and callable(reminder['callback']):
                reminder['callback'](
                    success=True, 
                    audio_path=f"{audio_path}.{self._get_audio_extension()}" if settings['save_audio'] else None
                )
                
        except Exception as e:
            logger.error(f"Error processing voice reminder: {e}")
            
            # Call callback with failure
            if 'callback' in reminder and callable(reminder['callback']):
                reminder['callback'](success=False, error=str(e))
    
    def _play_audio_file(self, audio_file):
        """Play an audio file using platform-specific method."""
        try:
            if self.platform == 'windows':
                # Use Windows Media Player
                os.startfile(audio_file)
            elif self.platform == 'macos':
                # Use macOS afplay
                subprocess.run(['afplay', audio_file])
            else:
                # Try various Linux players
                players = ['mpg123', 'mpg321', 'aplay', 'play']
                for player in players:
                    try:
                        subprocess.run([player, audio_file], 
                                      stdout=subprocess.DEVNULL, 
                                      stderr=subprocess.DEVNULL)
                        return  # Successfully played
                    except:
                        continue
                # If we get here, none of the players worked
                logger.warning(f"Could not find a suitable audio player on {self.platform}")
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
    
    def _create_placeholder_audio(self, filename):
        """Create a minimal placeholder audio file for web playback."""
        try:
            # Try to create a minimal MP3 file (1 second of silence)
            if 'gtts' in self.available_engines:
                from gtts import gTTS
                tts = gTTS(text=" ", lang='en')
                tts.save(filename)
            else:
                # Just create an empty file
                with open(filename, 'wb') as f:
                    f.write(b'')
        except Exception as e:
            logger.error(f"Error creating placeholder audio: {e}")
    
    def _get_audio_extension(self):
        """Get the appropriate audio file extension based on the engine."""
        if self.tts_engine in ['gtts', 'pyttsx3', 'web']:
            return 'mp3'
        elif self.tts_engine in ['espeak', 'festival', 'sapi']:
            return 'wav'
        else:
            return 'mp3'  # Default
    
    def queue_reminder(self, text, user_id=None, reminder_id=None, settings=None, callback=None):
        """
        Queue a voice reminder to be spoken.
        
        Args:
            text (str): The text of the reminder to be spoken
            user_id (int, optional): User ID associated with this reminder
            reminder_id (int, optional): Reminder ID in the database
            settings (dict, optional): Override default voice settings
            callback (callable, optional): Function to call after processing
            
        Returns:
            bool: True if reminder was queued successfully
        """
        try:
            reminder = {
                'text': text,
                'user_id': user_id,
                'reminder_id': reminder_id,
                'settings': settings or {},
                'callback': callback,
                'timestamp': datetime.now().isoformat()
            }
            
            self.voice_queue.put(reminder)
            logger.info(f"Queued voice reminder for user {user_id}: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error queueing voice reminder: {e}")
            return False
    
    def speak_reminder(self, text, user_id=None, reminder_id=None, settings=None):
        """
        Speak a reminder immediately (blocking).
        
        Args:
            text (str): The text of the reminder to be spoken
            user_id (int, optional): User ID associated with this reminder
            reminder_id (int, optional): Reminder ID in the database
            settings (dict, optional): Override default voice settings
            
        Returns:
            bool: True if reminder was spoken successfully
        """
        try:
            reminder = {
                'text': text,
                'user_id': user_id,
                'reminder_id': reminder_id,
                'settings': settings or {}
            }
            
            self._speak_reminder(reminder)
            return True
        except Exception as e:
            logger.error(f"Error speaking reminder: {e}")
            return False
    
    def get_available_voices(self):
        """
        Get a list of available voices for the current engine.
        
        Returns:
            list: List of available voice information
        """
        voices = []
        
        try:
            if self.tts_engine == 'pyttsx3' and hasattr(self, '_pyttsx3_engine'):
                pyttsx3_voices = self._pyttsx3_engine.getProperty('voices')
                for voice in pyttsx3_voices:
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'languages': voice.languages,
                        'gender': 'female' if 'female' in voice.name.lower() else 'male',
                        'engine': 'pyttsx3'
                    })
            elif self.tts_engine == 'sapi':
                import comtypes.client
                speaker = comtypes.client.CreateObject("SAPI.SpVoice")
                sapi_voices = speaker.GetVoices()
                
                for i in range(sapi_voices.Count):
                    voice = sapi_voices.Item(i)
                    desc = voice.GetDescription()
                    voices.append({
                        'id': str(voice.Id),
                        'name': desc,
                        'gender': 'female' if 'female' in desc.lower() else 'male',
                        'engine': 'sapi'
                    })
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
        
        return voices
    
    def get_audio_url(self, user_id, reminder_id, timestamp=None):
        """
        Get the URL for a reminder's audio file.
        
        Args:
            user_id (int): User ID
            reminder_id (int): Reminder ID
            timestamp (str, optional): Timestamp of the specific audio file
            
        Returns:
            str: URL or path to the audio file, or None if not found
        """
        try:
            # If timestamp is provided, look for that specific file
            if timestamp:
                file_name = f"reminder_{user_id}_{reminder_id}_{timestamp}"
                audio_path = os.path.join(self.audio_dir, f"{file_name}.{self._get_audio_extension()}")
                if os.path.exists(audio_path):
                    return audio_path
            
            # Otherwise, find the most recent file for this reminder
            pattern = f"reminder_{user_id}_{reminder_id}_"
            files = [f for f in os.listdir(self.audio_dir) 
                    if f.startswith(pattern) and 
                    f.endswith(f".{self._get_audio_extension()}")]
            
            if files:
                # Sort by timestamp (descending)
                files.sort(reverse=True)
                return os.path.join(self.audio_dir, files[0])
        
        except Exception as e:
            logger.error(f"Error getting audio URL: {e}")
        
        return None
    
    def stop(self):
        """Stop the voice reminder service."""
        logger.info("Stopping voice reminder service")
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            
        logger.info("Voice reminder service stopped")
        
    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.stop()


# Singleton instance for global access
_voice_reminder_service = None

def get_voice_reminder_service(config=None):
    """
    Get the global voice reminder service instance.
    
    Args:
        config (dict, optional): Configuration for the service
        
    Returns:
        VoiceReminderService: The voice reminder service instance
    """
    global _voice_reminder_service
    
    if _voice_reminder_service is None:
        _voice_reminder_service = VoiceReminderService(config)
    
    return _voice_reminder_service


def main():
    """Test the voice reminder service."""
    # Example config
    config = {
        'rate': 150,
        'volume': 1.0,
        'save_audio': True,
        'play_audio': True,
        'language': 'en-US'
    }
    
    # Initialize service
    service = get_voice_reminder_service(config)
    
    # Print available voices
    voices = service.get_available_voices()
    print(f"Available voices ({len(voices)}):")
    for i, voice in enumerate(voices):
        print(f"{i+1}. {voice['name']} ({voice['gender']})")
    
    # Test speaking
    service.speak_reminder(
        "This is a test reminder. The voice reminder service is working correctly.",
        user_id=1,
        reminder_id=123
    )
    
    # Test queueing
    service.queue_reminder(
        "This is a queued reminder. It should play after the first one.",
        user_id=1,
        reminder_id=124,
        callback=lambda success, **kwargs: print(f"Reminder callback: success={success}, {kwargs}")
    )
    
    # Wait for queued reminders to complete
    time.sleep(5)
    
    # Stop the service
    service.stop()
    
    print("Voice reminder service test completed.")


if __name__ == "__main__":
    main() 