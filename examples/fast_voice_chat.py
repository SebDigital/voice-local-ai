#!/usr/bin/env python3
"""
Fast Real-time Voice Chat using NeuTTS Air with streaming
Optimized for speed and better audio quality
"""

import argparse
import time
import threading
import queue
import speech_recognition as sr
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import neuttsair
sys.path.append(str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir


class FastVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf"):
        """Initialize the fast voice chat system"""
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
        print("Ready for fast voice chat!")
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Optimize recognition settings for speed
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5  # Shorter pause detection
        
        # Adjust for ambient noise
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Conversation context
        self.conversation_history = []
        
    def listen_for_speech(self, timeout=3):
        """Listen for speech and return transcribed text (optimized for speed)"""
        try:
            with self.microphone as source:
                print(f"\n🎤 Listening... (speak within {timeout} seconds)")
                # Shorter timeout and phrase limit for faster response
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
            
            print("🔄 Processing speech...")
            # Use faster recognition
            text = self.recognizer.recognize_google(audio)
            print(f"👤 You said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate a simple response to user input"""
        user_input_lower = user_input.lower()
        
        # Shorter, faster responses
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hi there! How can I help you?"
        
        elif any(question in user_input_lower for question in ['how are you', 'how do you do']):
            return "I'm doing great! Thanks for asking."
        
        elif any(question in user_input_lower for question in ['what', 'who', 'where', 'when', 'why', 'how']):
            return "That's interesting! Tell me more."
        
        elif any(word in user_input_lower for word in ['thank', 'thanks']):
            return "You're welcome!"
        
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'see you', 'quit', 'exit']):
            return "Goodbye! Have a great day!"
        
        elif any(word in user_input_lower for word in ['name', 'call']):
            return "I'm an AI assistant. Nice to meet you!"
        
        elif any(word in user_input_lower for word in ['time', 'clock']):
            current_time = time.strftime("%I:%M %p")
            return f"It's {current_time}."
        
        else:
            # Shorter default responses
            responses = [
                "That's interesting!",
                "Tell me more about that.",
                "I understand.",
                "Thanks for sharing!",
                "What else is on your mind?",
                "I'm listening.",
                "Please continue."
            ]
            import random
            return random.choice(responses)
    
    def synthesize_response(self, response_text):
        """Convert text response to speech using NeuTTS Air (optimized)"""
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Generating speech...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Normalize audio for better quality
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.8  # Normalize to 80% volume
            
            # Save and play audio with absolute path
            output_path = os.path.abspath("temp_response.wav")
            sf.write(output_path, wav, 24000)
            
            # Play audio (macOS) with error handling
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
        """Main conversation loop (optimized for speed)"""
        print("\n" + "="*60)
        print("🚀 FAST VOICE CHAT WITH NEUTTS AIR")
        print("="*60)
        print("Speak naturally and I'll respond quickly!")
        print("Say 'goodbye' or 'quit' to end the conversation.")
        print("="*60 + "\n")
        
        while True:
            # Listen for user input with shorter timeout
            user_input = self.listen_for_speech(timeout=3)
            
            if user_input is None:
                continue
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'quit', 'exit', 'stop']):
                self.synthesize_response("Goodbye! Have a great day!")
                break
            
            # Add to conversation history
            self.conversation_history.append(("user", user_input))
            
            # Generate response
            response = self.generate_response(user_input)
            self.conversation_history.append(("assistant", response))
            
            # Synthesize and speak response
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Fast Real-time Voice Chat with NeuTTS Air")
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
        # Initialize voice chat
        chat = FastVoiceChat(args.ref_audio, args.ref_text, args.backbone)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
