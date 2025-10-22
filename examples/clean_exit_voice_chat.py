#!/usr/bin/env python3
"""
Clean Exit Voice Chat - Properly handles stopping and cleanup
"""

import argparse
import os
import signal
import sys
import tempfile
from pathlib import Path

import speech_recognition as sr
from neuttsair.neutts import NeuTTSAir


class CleanExitVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf", whisper_model="base"):
        self.running = True
        
        # Setup signal handlers for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 Initializing Clean Exit Voice Chat...")
        
        # Initialize TTS
        print("🎵 Loading NeuTTS Air...")
        self.tts = NeuTTSAir(backbone=backbone)
        
        # Load reference audio and text
        print(f"📁 Loading reference audio: {ref_audio_path}")
        print(f"📁 Loading reference text: {ref_text_path}")
        
        with open(ref_text_path, 'r') as f:
            self.ref_text = f.read().strip()
        
        # Encode reference for faster inference
        import soundfile as sf
        ref_audio, _ = sf.read(ref_audio_path)
        self.ref_codes = self.tts.encode(ref_audio, self.ref_text)
        
        # Setup speech recognition
        print("🎤 Setting up speech recognition...")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        print("🔧 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Setup Whisper if requested
        self.use_whisper = True
        if self.use_whisper:
            print(f"🤖 Loading Whisper model: {whisper_model}")
            import whisper
            self.whisper_model_obj = whisper.load_model(whisper_model)
            print("✅ Whisper loaded successfully!")
        
        # Conversation context
        self.conversation_history = []
        
        print("✅ Initialization complete!")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals for clean exit"""
        print(f"\n\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        try:
            # Kill any running audio processes
            os.system("pkill -f afplay 2>/dev/null || true")
            os.system("pkill -f aplay 2>/dev/null || true")
            
            # Clean up temporary files
            temp_dir = tempfile.gettempdir()
            for file in os.listdir(temp_dir):
                if file.startswith("temp_response_") and file.endswith(".wav"):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
            
            print("✅ Cleanup complete!")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def listen_for_speech(self, timeout=5):
        """Listen for speech with proper error handling"""
        if not self.running:
            return None
            
        try:
            with self.microphone as source:
                print(f"\n🎤 Listening... (speak within {timeout} seconds)")
                print("💡 Press Ctrl+C to stop at any time")
                
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            if not self.running:
                return None
                
            print("🔄 Processing speech...")
            
            if self.use_whisper:
                return self.process_with_whisper(audio)
            else:
                return self.process_with_google(audio)
                
        except sr.WaitTimeoutError:
            if self.running:
                print("⏰ Listening timeout - no speech detected")
            return None
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.running = False
            return None
        except Exception as e:
            if self.running:
                print(f"❌ Speech recognition error: {e}")
            return None
    
    def process_with_whisper(self, audio):
        """Process audio with Whisper"""
        try:
            import io
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav_data = audio.get_wav_data()
                tmp_file.write(wav_data)
                tmp_file_path = tmp_file.name
            
            try:
                if not self.running:
                    return None
                    
                # Transcribe with Whisper
                result = self.whisper_model_obj.transcribe(tmp_file_path)
                text = result["text"].strip()
                
                if text and self.running:
                    print(f"👤 You said: {text}")
                    return text
                else:
                    print("❓ No speech detected")
                    return None
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Whisper processing error: {e}")
            return None
    
    def process_with_google(self, audio):
        """Process audio with Google Speech Recognition"""
        try:
            text = self.recognizer.recognize_google(audio)
            if text and self.running:
                print(f"👤 You said: {text}")
                return text
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate intelligent responses"""
        # Add user input to history
        self.conversation_history.append(("user", user_input))
        
        # Simple response generation with context awareness
        user_lower = user_input.lower()
        
        # Check for greetings
        if any(greeting in user_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            responses = [
                f"Hello! Great to meet you! I'm your local AI assistant.",
                f"Hi there! I'm running completely offline on your device.",
                f"Hey! Nice to chat with you. What would you like to talk about?"
            ]
            response = responses[len(self.conversation_history) % len(responses)]
        
        # Check for questions about capabilities
        elif any(word in user_lower for word in ['what can you do', 'capabilities', 'help', 'what are you']):
            response = "I'm a privacy-focused voice assistant running entirely on your device. I can have conversations, answer questions, and help with various tasks - all while keeping your data completely private!"
        
        # Check for privacy questions
        elif any(word in user_lower for word in ['privacy', 'data', 'cloud', 'offline']):
            response = "That's my specialty! Everything we discuss stays right here on your device. No data goes to the cloud, no companies can access our conversation. It's complete privacy!"
        
        # Check for technology questions
        elif any(word in user_lower for word in ['how', 'technology', 'whisper', 'neutts', 'voice']):
            response = "I use OpenAI Whisper for speech recognition and NeuTTS Air for voice synthesis. Both run locally on your device, giving you powerful AI without compromising your privacy."
        
        # Check for goodbye
        elif any(word in user_lower for word in ['goodbye', 'bye', 'see you', 'farewell']):
            response = "Goodbye! It was wonderful talking with you. Remember, everything we discussed stayed private on your device. Take care!"
        
        # General responses with context
        else:
            # Provide varied, contextual responses
            responses = [
                f"That's interesting! Tell me more about {user_input.split()[-1] if user_input.split() else 'that'}.",
                f"I understand. From a privacy perspective, that's really important.",
                f"Thanks for sharing that with me. What else would you like to discuss?",
                f"That's a great point. Since we're talking locally, I can be completely honest with you.",
                f"Fascinating! I'm processing this entirely on your device, so our conversation is completely private."
            ]
            response = responses[len(self.conversation_history) % len(responses)]
        
        # Add response to history
        self.conversation_history.append(("assistant", response))
        
        return response
    
    def synthesize_response(self, response_text):
        """Convert text response to speech using NeuTTS Air"""
        if not self.running:
            return
            
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Generating speech...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Create unique temporary file
            import time
            timestamp = int(time.time() * 1000)
            output_path = os.path.abspath(f"temp_response_{timestamp}.wav")
            
            # Save audio
            import soundfile as sf
            sf.write(output_path, wav, 24000)
            
            if not self.running:
                return
            
            # Play audio (macOS) with error handling
            import subprocess
            result = subprocess.run(["afplay", output_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Audio playback error: {result.stderr}")
            
            # Clean up
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
    
    def run_conversation(self):
        """Main conversation loop with proper exit handling"""
        print("\n" + "="*70)
        print("🚀 CLEAN EXIT VOICE CHAT - 100% OFFLINE")
        print("="*70)
        if self.use_whisper:
            print("✅ Whisper Speech Recognition (offline)")
        else:
            print("⚠️ Google Speech Recognition (requires internet)")
        print("✅ NeuTTS Air Voice Synthesis (offline)")
        print("🔒 Complete Privacy - Your data stays on your device")
        print("💡 Say 'goodbye' or 'quit' to end")
        print("🛑 Press Ctrl+C at any time for clean exit")
        print("="*70 + "\n")
        
        try:
            while self.running:
                # Listen for user input
                user_input = self.listen_for_speech(timeout=5)
                
                if not self.running:
                    break
                
                if user_input is None:
                    continue
                
                # Check for exit commands
                if any(word in user_input.lower() for word in ['goodbye', 'quit', 'exit', 'stop', 'bye']):
                    self.synthesize_response("Goodbye! Thanks for using the clean exit voice chat. Everything processed locally for your privacy!")
                    break
                
                # Generate response
                response = self.generate_response(user_input)
                
                # Synthesize and speak response
                self.synthesize_response(response)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            print("\n🧹 Cleaning up...")
            self.cleanup()
            print("👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Clean Exit Voice Chat - Properly handles stopping and cleanup")
    parser.add_argument("--ref_audio", default="samples/dave.wav", 
                       help="Reference audio file for voice cloning")
    parser.add_argument("--ref_text", default="samples/dave.txt", 
                       help="Reference text file for voice cloning")
    parser.add_argument("--backbone", default="neuphonic/neutts-air-q4-gguf",
                       help="Backbone model to use")
    parser.add_argument("--whisper_model", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (tiny=fastest, large=most accurate)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.ref_audio).exists():
        print(f"❌ Reference audio file not found: {args.ref_audio}")
        return
    
    if not Path(args.ref_text).exists():
        print(f"❌ Reference text file not found: {args.ref_text}")
        return
    
    try:
        # Initialize clean exit voice chat
        chat = CleanExitVoiceChat(args.ref_audio, args.ref_text, args.backbone, args.whisper_model)
        
        # Start conversation
        chat.run_conversation()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🏁 Program ended")


if __name__ == "__main__":
    main()
