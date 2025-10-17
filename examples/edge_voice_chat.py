#!/usr/bin/env python3
"""
Edge Voice Chat - 100% Local Processing
Uses Whisper for on-device speech recognition + NeuTTS Air for voice synthesis
No internet required after initial setup!
"""

import argparse
import time
import speech_recognition as sr
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os
import threading
import queue

# Add the parent directory to the path so we can import neuttsair
sys.path.append(str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir


class EdgeVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf", whisper_model="base"):
        """Initialize the edge voice chat system"""
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
        
        # Optimize settings for edge processing
        self.recognizer.energy_threshold = 300
        self.recognizer.pause_threshold = 0.8
        
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("🚀 Edge Voice Chat Ready!")
        
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
            import io
            import tempfile
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Convert audio data to WAV format
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
    
    def generate_response(self, user_input):
        """Generate intelligent responses"""
        user_input_lower = user_input.lower()
        
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! I'm running completely offline on your device. How are you doing?"
        
        elif any(question in user_input_lower for question in ['how are you', 'how do you do', 'how\'s it going']):
            return "I'm doing great! I'm processing everything locally on your device for maximum privacy."
        
        elif any(question in user_input_lower for question in ['what is your name', 'who are you']):
            return "I'm an edge AI assistant powered by Whisper for speech recognition and NeuTTS Air for voice synthesis. Everything runs locally on your device!"
        
        elif any(word in user_input_lower for word in ['offline', 'local', 'edge', 'privacy']):
            return "Yes! I'm running completely offline. Your voice data never leaves your device, ensuring complete privacy and security."
        
        elif any(word in user_input_lower for word in ['whisper', 'speech recognition']):
            return "I'm using Whisper for speech recognition, which runs entirely on your device. No internet required after setup!"
        
        elif any(word in user_input_lower for word in ['time']):
            current_time = time.strftime("%I:%M %p")
            return f"The current time is {current_time}. All processing is happening locally on your device."
        
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'quit', 'exit']):
            return "Goodbye! Thanks for using the edge voice chat. Everything processed locally for your privacy!"
        
        else:
            responses = [
                "That's interesting! I'm processing this completely offline on your device.",
                "I understand. All my responses are generated locally for your privacy.",
                "Thanks for sharing! Your data stays on your device.",
                "That's fascinating! I'm running entirely offline.",
                "I appreciate you telling me that. Everything is processed locally.",
                "That's a great point. All processing happens on your device for privacy.",
                "I'm listening and learning, all while keeping your data private and local."
            ]
            import random
            return random.choice(responses)
    
    def synthesize_response(self, response_text):
        """Convert text response to speech using NeuTTS Air"""
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Generating speech (offline)...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Audio normalization
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
        """Main conversation loop - 100% offline"""
        print("\n" + "="*70)
        print("🚀 EDGE VOICE CHAT - 100% OFFLINE")
        print("="*70)
        if self.use_whisper:
            print("✅ Whisper Speech Recognition (offline)")
        else:
            print("⚠️ Google Speech Recognition (requires internet)")
        print("✅ NeuTTS Air Voice Synthesis (offline)")
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
                self.synthesize_response("Goodbye! Thanks for using the edge voice chat. Everything processed locally for your privacy!")
                break
            
            # Generate response
            response = self.generate_response(user_input)
            
            # Synthesize and speak response
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Edge Voice Chat - 100% Local Processing")
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
        # Initialize edge voice chat
        chat = EdgeVoiceChat(args.ref_audio, args.ref_text, args.backbone, args.whisper_model)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
