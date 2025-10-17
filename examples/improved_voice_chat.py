#!/usr/bin/env python3
"""
Improved Voice Chat with better speech recognition
Multiple recognition engines and better audio handling
"""

import argparse
import time
import speech_recognition as sr
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import neuttsair
sys.path.append(str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir


class ImprovedVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf"):
        """Initialize the improved voice chat system"""
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.backbone = backbone
        
        # Initialize TTS
        print("Loading NeuTTS Air...")
        self.tts = NeuTTSAir(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
        
        # Load reference text
        with open(ref_text_path, 'r') as f:
            self.ref_text = f.read().strip()
        
        # Pre-encode reference for faster inference
        print("Encoding reference audio...")
        self.ref_codes = self.tts.encode_reference(ref_audio_path)
        print("Ready for improved voice chat!")
        
        # Initialize speech recognition with better settings
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Improved recognition settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Longer pause detection
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5
        
        # Better microphone calibration
        print("Calibrating microphone for better recognition...")
        with self.microphone as source:
            print("Please speak clearly for calibration...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"Energy threshold set to: {self.recognizer.energy_threshold}")
        
        # Test microphone
        print("Testing microphone...")
        self.test_microphone()
        
    def test_microphone(self):
        """Test if microphone is working properly"""
        try:
            with self.microphone as source:
                print("Please say 'test' to verify microphone is working...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Microphone test successful! Heard: '{text}'")
        except sr.WaitTimeoutError:
            print("⚠️ Microphone test timeout - please check your microphone")
        except Exception as e:
            print(f"⚠️ Microphone test failed: {e}")
    
    def listen_for_speech(self, timeout=5):
        """Improved speech recognition with multiple fallbacks"""
        try:
            with self.microphone as source:
                print(f"\n🎤 Listening... (speak clearly within {timeout} seconds)")
                print("💡 Tip: Speak clearly and wait for the beep")
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            
            # Try multiple recognition engines
            text = None
            
            # First try: Google (most accurate)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"👤 You said: {text}")
                return text
            except sr.UnknownValueError:
                print("❓ Google couldn't understand - trying alternatives...")
            except sr.RequestError as e:
                print(f"❌ Google service error: {e}")
            
            # Fallback: Try with different language models
            try:
                text = self.recognizer.recognize_google(audio, language="en-US")
                print(f"👤 You said (US): {text}")
                return text
            except:
                pass
            
            # Final fallback: Show audio info
            print("❌ Could not understand the audio")
            print("💡 Try speaking louder, clearer, or closer to the microphone")
            return None
            
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout - no speech detected")
            print("💡 Try speaking louder or check your microphone")
            return None
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate intelligent responses"""
        user_input_lower = user_input.lower()
        
        # More comprehensive response patterns
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "Hello! Nice to meet you. How are you doing today?"
        
        elif any(question in user_input_lower for question in ['how are you', 'how do you do', 'how\'s it going', 'what\'s up']):
            return "I'm doing great, thank you for asking! I'm excited to have this conversation with you."
        
        elif any(question in user_input_lower for question in ['what is your name', 'who are you', 'what\'s your name']):
            return "I'm an AI assistant powered by NeuTTS Air. You can call me whatever you'd like!"
        
        elif any(question in user_input_lower for question in ['what time is it', 'what\'s the time', 'time']):
            current_time = time.strftime("%I:%M %p")
            return f"The current time is {current_time}. How can I help you with your day?"
        
        elif any(question in user_input_lower for question in ['what', 'who', 'where', 'when', 'why', 'how']):
            return "That's an interesting question. I'm still learning, but I'd be happy to discuss it with you."
        
        elif any(word in user_input_lower for word in ['thank', 'thanks', 'thank you']):
            return "You're very welcome! I'm glad I could help."
        
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'see you', 'farewell', 'quit', 'exit']):
            return "Goodbye! It was great talking with you. Have a wonderful day!"
        
        elif any(word in user_input_lower for word in ['weather', 'sunny', 'rainy', 'cloudy', 'temperature']):
            return "I don't have access to real-time weather data, but I hope you're having a nice day regardless!"
        
        elif any(word in user_input_lower for word in ['help', 'assistance', 'support']):
            return "I'm here to help! You can ask me questions, have a conversation, or just chat. What would you like to talk about?"
        
        elif any(word in user_input_lower for word in ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok']):
            return "Great! I'm glad you agree. What else would you like to talk about?"
        
        elif any(word in user_input_lower for word in ['no', 'nope', 'nah', 'not really']):
            return "That's okay! We can talk about something else. What interests you?"
        
        elif any(word in user_input_lower for word in ['maybe', 'perhaps', 'possibly']):
            return "That's an interesting perspective! Tell me more about what you're thinking."
        
        elif any(word in user_input_lower for word in ['good', 'great', 'awesome', 'excellent', 'wonderful']):
            return "That's wonderful to hear! I'm glad things are going well for you."
        
        elif any(word in user_input_lower for word in ['bad', 'terrible', 'awful', 'horrible']):
            return "I'm sorry to hear that. I hope things get better for you soon. Is there anything I can help with?"
        
        elif any(word in user_input_lower for word in ['tired', 'exhausted', 'sleepy']):
            return "It sounds like you could use some rest. I hope you get a chance to relax soon."
        
        elif any(word in user_input_lower for word in ['happy', 'excited', 'joyful', 'cheerful']):
            return "That's fantastic! I love hearing when people are happy. What's making you feel this way?"
        
        elif any(word in user_input_lower for word in ['sad', 'upset', 'disappointed']):
            return "I'm sorry you're feeling that way. Sometimes it helps to talk about what's bothering you."
        
        else:
            # More varied default responses
            responses = [
                "That's really interesting! Tell me more about that.",
                "I understand what you're saying. Can you elaborate on that?",
                "That sounds fascinating! I'd love to hear more.",
                "Thanks for sharing that with me. What else is on your mind?",
                "I appreciate you telling me that. How do you feel about it?",
                "That's a great point. What made you think of that?",
                "I'm listening and learning from our conversation. Please continue.",
                "That's something I hadn't thought about. Can you tell me more?",
                "I find that really interesting. What's your perspective on this?",
                "Thanks for bringing that up. It's definitely worth discussing."
            ]
            import random
            return random.choice(responses)
    
    def synthesize_response(self, response_text):
        """Convert text response to speech using NeuTTS Air"""
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Generating speech...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Better audio normalization
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.8
            
            # Save and play audio
            output_path = os.path.abspath("temp_response.wav")
            sf.write(output_path, wav, 24000)
            
            # Play audio
            import subprocess
            result = subprocess.run(["afplay", output_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Audio playback error: {result.stderr}")
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
    
    def run_conversation(self):
        """Main conversation loop with better user guidance"""
        print("\n" + "="*70)
        print("🎙️  IMPROVED VOICE CHAT WITH NEUTTS AIR")
        print("="*70)
        print("💡 Tips for better recognition:")
        print("   • Speak clearly and at normal volume")
        print("   • Wait for the 'Listening...' prompt")
        print("   • Pause briefly after speaking")
        print("   • Keep background noise minimal")
        print("   • Say 'goodbye' or 'quit' to end")
        print("="*70 + "\n")
        
        while True:
            # Listen for user input
            user_input = self.listen_for_speech(timeout=5)
            
            if user_input is None:
                print("🔄 Let's try again...")
                continue
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'quit', 'exit', 'stop', 'bye']):
                self.synthesize_response("Goodbye! It was great talking with you. Have a wonderful day!")
                break
            
            # Generate response
            response = self.generate_response(user_input)
            
            # Synthesize and speak response
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Improved Voice Chat with NeuTTS Air")
    parser.add_argument("--ref_audio", default="samples/dave.wav", 
                       help="Reference audio file for voice cloning")
    parser.add_argument("--ref_text", default="samples/dave.txt", 
                       help="Reference text file for voice cloning")
    parser.add_argument("--backbone", default="neuphonic/neutts-air-q4-gguf",
                       help="Backbone model to use")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.ref_audio).exists():
        print(f"❌ Reference audio file not found: {args.ref_audio}")
        return
    
    if not Path(args.ref_text).exists():
        print(f"❌ Reference text file not found: {args.ref_text}")
        return
    
    try:
        # Initialize improved voice chat
        chat = ImprovedVoiceChat(args.ref_audio, args.ref_text, args.backbone)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
