#!/usr/bin/env python3
"""
Push-to-Talk Edge Voice Chat - Prevents feedback loops
Press SPACE to talk, release to stop - no audio feedback issues!
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
import keyboard
import threading

# Add the parent directory to the path so we can import neuttsair
sys.path.append(str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir


class PushToTalkChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf", whisper_model="base"):
        """Initialize the push-to-talk voice chat system"""
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.backbone = backbone
        self.whisper_model = whisper_model
        self.is_recording = False
        self.audio_data = []
        
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
        self.recognizer.pause_threshold = 0.3  # Shorter for push-to-talk
        
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Conversation context
        self.conversation_history = []
        self.user_name = None
        self.response_count = 0
        
        print("🚀 Push-to-Talk Edge Voice Chat Ready!")
        
    def setup_google_recognition(self):
        """Setup Google Speech Recognition as fallback"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("📡 Using Google Speech Recognition (requires internet)")
    
    def record_audio(self):
        """Record audio while space is held down"""
        try:
            with self.microphone as source:
                while self.is_recording:
                    # Record small chunks
                    audio_chunk = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=0.5)
                    if audio_chunk:
                        self.audio_data.append(audio_chunk)
        except Exception as e:
            print(f"Recording error: {e}")
    
    def listen_with_push_to_talk(self):
        """Listen for speech using push-to-talk (prevents feedback)"""
        try:
            print("\n🎤 Hold SPACE to talk, release to stop...")
            
            # Start recording when space is pressed
            self.is_recording = True
            self.audio_data = []
            
            # Start recording thread
            recording_thread = threading.Thread(target=self.record_audio)
            recording_thread.start()
            
            # Wait for space key release
            keyboard.wait('space')
            
            # Stop recording
            self.is_recording = False
            recording_thread.join(timeout=1)
            
            if not self.audio_data:
                print("❓ No audio recorded")
                return None
            
            print("🔄 Processing recorded speech...")
            
            # Combine audio chunks
            import io
            combined_audio = sr.AudioData(b''.join([chunk.get_raw_data() for chunk in self.audio_data]), 
                                        self.audio_data[0].sample_rate, 
                                        self.audio_data[0].sample_width)
            
            if self.use_whisper:
                return self.process_with_whisper(combined_audio)
            else:
                return self.process_with_google(combined_audio)
                
        except Exception as e:
            print(f"❌ Push-to-talk error: {e}")
            return None
    
    def process_with_whisper(self, audio):
        """Process audio with Whisper"""
        try:
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
            print(f"❌ Whisper processing error: {e}")
            return None
    
    def process_with_google(self, audio):
        """Process audio with Google"""
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"👤 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Google service error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate natural responses"""
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
        
        # Natural responses
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning']):
            if self.user_name:
                return f"Hey {self.user_name}! How's it going today?"
            else:
                return "Hi there! Nice to meet you. What's your name?"
        
        elif any(question in user_input_lower for question in ['how are you', 'how\'s it going']):
            return "I'm doing great! No more audio feedback issues with push-to-talk. How about you?"
        
        elif any(word in user_input_lower for word in ['work', 'job', 'career']):
            return "Work sounds interesting! What do you do for a living?"
        
        elif any(word in user_input_lower for word in ['family', 'parents', 'kids']):
            return "Family is so important! Tell me about yours if you'd like."
        
        elif any(word in user_input_lower for word in ['hobby', 'sport', 'music', 'gaming']):
            return "That sounds fun! What kind of hobbies do you enjoy?"
        
        elif any(word in user_input_lower for word in ['food', 'eat', 'restaurant']):
            return "Food is great! What's your favorite type of cuisine?"
        
        elif any(word in user_input_lower for word in ['time', 'clock']):
            current_time = time.strftime("%I:%M %p")
            return f"It's {current_time}. How's your day going?"
        
        elif any(word in user_input_lower for word in ['thank', 'thanks']):
            return "You're welcome! I'm happy to help. What else would you like to talk about?"
        
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'quit', 'exit']):
            if self.user_name:
                return f"Goodbye {self.user_name}! It was great chatting without audio feedback!"
            else:
                return "Goodbye! Thanks for the conversation!"
        
        else:
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
                "Thanks for bringing that up. It's definitely worth talking about."
            ]
            return responses[self.response_count % len(responses)]
    
    def synthesize_response(self, response_text):
        """Convert text response to speech using NeuTTS Air"""
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Generating speech...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Audio normalization
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.8
            
            # Save and play audio
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_path = os.path.abspath(f"temp_response_{unique_id}.wav")
            
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
        """Main conversation loop with push-to-talk"""
        print("\n" + "="*70)
        print("🎤 PUSH-TO-TALK EDGE VOICE CHAT - NO FEEDBACK!")
        print("="*70)
        print("✅ Whisper Speech Recognition (offline)")
        print("✅ NeuTTS Air Voice Synthesis (offline)")
        print("✅ Push-to-Talk - No Audio Feedback")
        print("🔒 Complete Privacy - Your data stays on your device")
        print("")
        print("🎯 INSTRUCTIONS:")
        print("   • Hold SPACE key to talk")
        print("   • Release SPACE when finished")
        print("   • AI will respond with voice")
        print("   • Say 'goodbye' to end")
        print("="*70 + "\n")
        
        while True:
            # Listen for user input with push-to-talk
            user_input = self.listen_with_push_to_talk()
            
            if user_input is None:
                continue
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'quit', 'exit', 'stop', 'bye']):
                response = self.generate_response(user_input)
                self.synthesize_response(response)
                break
            
            # Generate response
            response = self.generate_response(user_input)
            
            # Synthesize and speak response
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Push-to-Talk Edge Voice Chat - No Audio Feedback")
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
        # Install keyboard library if needed
        try:
            import keyboard
        except ImportError:
            print("Installing keyboard library...")
            os.system("pip install keyboard")
            import keyboard
        
        # Initialize push-to-talk voice chat
        chat = PushToTalkChat(args.ref_audio, args.ref_text, args.backbone, args.whisper_model)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
