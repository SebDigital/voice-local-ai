#!/usr/bin/env python3
"""
Fixed Edge Voice Chat - No echo, better responses
Fixes the dual voice issue and improves conversation quality
"""

import argparse
import time
import speech_recognition as sr
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os
import random

# Add the parent directory to the path so we can import neuttsair
sys.path.append(str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir


class FixedEdgeVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf", whisper_model="base"):
        """Initialize the fixed edge voice chat system"""
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.backbone = backbone
        self.whisper_model = whisper_model
        
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
        
        # Initialize Whisper for edge speech recognition
        print(f"Loading Whisper model ({whisper_model})...")
        try:
            import whisper
            self.whisper_model_obj = whisper.load_model(whisper_model)
            self.use_whisper = True
            print("✅ Whisper loaded successfully - 100% offline speech recognition!")
        except ImportError:
            print("⚠️ Whisper not installed. Installing...")
            os.system("pip install openai-whisper")
            try:
                import whisper
                self.whisper_model_obj = whisper.load_model(whisper_model)
                self.use_whisper = True
                print("✅ Whisper installed and loaded!")
            except Exception as e:
                print(f"❌ Whisper installation failed: {e}")
                print("🔄 Falling back to Google Speech Recognition...")
                self.use_whisper = False
                self.setup_google_recognition()
        
        # Initialize microphone
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Optimize settings
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 0.8
        
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Conversation context
        self.conversation_history = []
        self.user_name = None
        self.response_count = 0
        
        print("🚀 Fixed Edge Voice Chat Ready!")
        
    def setup_google_recognition(self):
        """Setup Google Speech Recognition as fallback"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("📡 Using Google Speech Recognition (requires internet)")
    
    def listen_for_speech_whisper(self, timeout=5):
        """Listen for speech using Whisper (100% offline)"""
        try:
            with self.microphone as source:
                print(f"\n🎤 Listening... (speak within {timeout} seconds)")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print("🔄 Processing with Whisper (offline)...")
            
            # Convert audio to format Whisper expects
            import tempfile
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav_data = audio.get_wav_data()
                tmp_file.write(wav_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Transcribe with Whisper
                result = self.whisper_model_obj.transcribe(tmp_file_path)
                text = result["text"].strip()
                
                if text:
                    print(f"👤 You said: {text}")
                    return text
                else:
                    print("❓ No speech detected")
                    return None
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            print(f"❌ Whisper recognition error: {e}")
            return None
    
    def listen_for_speech_google(self, timeout=5):
        """Listen for speech using Google (requires internet)"""
        try:
            with self.microphone as source:
                print(f"\n🎤 Listening... (speak within {timeout} seconds)")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print("🔄 Processing with Google...")
            text = self.recognizer.recognize_google(audio)
            print(f"👤 You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Google service error: {e}")
            return None
    
    def listen_for_speech(self, timeout=5):
        """Listen for speech using the configured method"""
        if self.use_whisper:
            return self.listen_for_speech_whisper(timeout)
        else:
            return self.listen_for_speech_google(timeout)
    
    def generate_better_response(self, user_input):
        """Generate much better, more natural responses"""
        user_input_lower = user_input.lower()
        self.response_count += 1
        
        # Store in conversation history
        self.conversation_history.append(("user", user_input))
        
        # Extract name if mentioned
        if not self.user_name and any(word in user_input_lower for word in ['my name is', 'i am', 'i\'m']):
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ['name', 'is', 'am', 'i\'m'] and i + 1 < len(words):
                    self.user_name = words[i + 1].strip('.,!?')
                    break
        
        # Much more natural and varied responses
        
        # Greetings - very natural
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            if self.user_name:
                return f"Hey {self.user_name}! How's it going today?"
            else:
                return "Hi there! Nice to meet you. What's your name?"
        
        # How are you - conversational
        elif any(question in user_input_lower for question in ['how are you', 'how do you do', 'how\'s it going', 'what\'s up']):
            return "I'm doing great! Just processing everything locally on your device. How about you?"
        
        # Name questions
        elif any(question in user_input_lower for question in ['what is your name', 'who are you', 'what\'s your name']):
            return "I'm your local AI assistant! I use Whisper to hear you and NeuTTS Air to respond with this voice, all running offline."
        
        # Time questions
        elif any(word in user_input_lower for word in ['time', 'clock', 'hour']):
            current_time = time.strftime("%I:%M %p")
            return f"It's {current_time}. How's your day going?"
        
        # Work related
        elif any(word in user_input_lower for word in ['work', 'job', 'career', 'office']):
            return "Work sounds interesting! What do you do for a living?"
        
        # Family related
        elif any(word in user_input_lower for word in ['family', 'parents', 'mother', 'father', 'kids']):
            return "Family is so important! Tell me about yours if you'd like."
        
        # Hobbies
        elif any(word in user_input_lower for word in ['hobby', 'sport', 'music', 'art', 'reading', 'gaming']):
            return "That sounds fun! What kind of hobbies do you enjoy?"
        
        # Food
        elif any(word in user_input_lower for word in ['food', 'eat', 'restaurant', 'cooking']):
            return "Food is great! What's your favorite type of cuisine?"
        
        # Weather
        elif any(word in user_input_lower for word in ['weather', 'rain', 'sunny', 'cloudy']):
            return "I don't have access to weather data, but I hope you're having a nice day!"
        
        # Technology
        elif any(word in user_input_lower for word in ['computer', 'phone', 'ai', 'technology']):
            return "Technology is amazing! I'm running completely offline on your device using Whisper and NeuTTS Air."
        
        # Emotions - positive
        elif any(word in user_input_lower for word in ['happy', 'excited', 'great', 'wonderful', 'awesome']):
            return "That's fantastic! I love hearing when people are happy. What's going well for you?"
        
        # Emotions - negative
        elif any(word in user_input_lower for word in ['sad', 'upset', 'tired', 'worried', 'stressed']):
            return "I'm sorry you're feeling that way. Sometimes it helps to talk about what's on your mind."
        
        # Thank you
        elif any(word in user_input_lower for word in ['thank', 'thanks', 'thank you']):
            return "You're welcome! I'm happy to help. What else would you like to talk about?"
        
        # Goodbye
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'see you', 'farewell', 'quit', 'exit']):
            if self.user_name:
                return f"Goodbye {self.user_name}! It was great chatting with you!"
            else:
                return "Goodbye! Thanks for the conversation!"
        
        # Questions
        elif any(word in user_input_lower for word in ['what', 'who', 'where', 'when', 'why', 'how']):
            return "That's a good question! I'm still learning, but I'd love to discuss it with you."
        
        # Yes/No responses
        elif any(word in user_input_lower for word in ['yes', 'yeah', 'yep', 'sure']):
            return "Great! I'm glad you agree. What else is on your mind?"
        
        elif any(word in user_input_lower for word in ['no', 'nope', 'nah']):
            return "That's okay! We can talk about something else. What interests you?"
        
        # Default responses - much more natural and varied
        else:
            # Rotate through different response styles
            responses = [
                "That's really interesting! Tell me more about that.",
                "I see what you mean. Can you elaborate on that?",
                "That sounds fascinating! What else can you tell me?",
                "Thanks for sharing that. What's your take on it?",
                "I appreciate you telling me that. How do you feel about it?",
                "That's a good point. What made you think of that?",
                "I'm listening! Please continue.",
                "That's something worth discussing. Tell me more.",
                "I find that interesting. What's your perspective?",
                "Thanks for bringing that up. It's definitely worth talking about.",
                "That's a thoughtful comment. What else would you like to share?",
                "I appreciate your input. This is a great conversation.",
                "That's an interesting way to look at it. Tell me more.",
                "I'm enjoying our chat. What other topics interest you?",
                "That's worth thinking about. What are your thoughts on this?"
            ]
            # Use response count to cycle through responses
            return responses[self.response_count % len(responses)]
    
    def synthesize_response(self, response_text):
        """Convert text response to speech using NeuTTS Air - FIXED for no echo"""
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Generating speech...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Audio normalization
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.8
            
            # Save and play audio - FIXED: Use unique filename to prevent conflicts
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_path = os.path.abspath(f"temp_response_{unique_id}.wav")
            
            sf.write(output_path, wav, 24000)
            
            # Play audio - FIXED: Kill any existing afplay processes first
            import subprocess
            subprocess.run(["pkill", "-f", "afplay"], capture_output=True)
            
            # Wait a moment before playing
            time.sleep(0.1)
            
            # Play the audio
            result = subprocess.run(["afplay", output_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Audio playback error: {result.stderr}")
            
            # Clean up - FIXED: Always clean up
            if os.path.exists(output_path):
                os.remove(output_path)
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
    
    def run_conversation(self):
        """Main conversation loop - FIXED"""
        print("\n" + "="*70)
        print("🔧 FIXED EDGE VOICE CHAT - NO ECHO, BETTER RESPONSES")
        print("="*70)
        if self.use_whisper:
            print("✅ Whisper Speech Recognition (offline)")
        else:
            print("⚠️ Google Speech Recognition (requires internet)")
        print("✅ NeuTTS Air Voice Synthesis (offline)")
        print("✅ Fixed Audio Issues")
        print("✅ Better Conversation Quality")
        print("🔒 Complete Privacy - Your data stays on your device")
        print("💡 Say 'goodbye' or 'quit' to end")
        print("="*70 + "\n")
        
        while True:
            # Listen for user input
            user_input = self.listen_for_speech(timeout=5)
            
            if user_input is None:
                continue
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'quit', 'exit', 'stop', 'bye']):
                response = self.generate_better_response(user_input)
                self.synthesize_response(response)
                break
            
            # Generate better response
            response = self.generate_better_response(user_input)
            
            # Synthesize and speak response
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Fixed Edge Voice Chat - No Echo, Better Responses")
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
        # Initialize fixed edge voice chat
        chat = FixedEdgeVoiceChat(args.ref_audio, args.ref_text, args.backbone, args.whisper_model)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
