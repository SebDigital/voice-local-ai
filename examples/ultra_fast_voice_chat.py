#!/usr/bin/env python3
"""
Ultra Fast Voice Chat using NeuTTS Air with ONNX decoder
Maximum speed and minimal latency
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


class UltraFastVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf"):
        """Initialize the ultra fast voice chat system"""
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.backbone = backbone
        
        # Initialize TTS with ONNX decoder for speed
        print("Loading NeuTTS Air with ONNX decoder...")
        self.tts = NeuTTSAir(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec-onnx-decoder",  # Use ONNX decoder
            codec_device="cpu"
        )
        
        # Load reference text
        with open(ref_text_path, 'r') as f:
            self.ref_text = f.read().strip()
        
        # Pre-encode reference for faster inference
        print("Encoding reference audio...")
        self.ref_codes = self.tts.encode_reference(ref_audio_path)
        print("Ready for ultra fast voice chat!")
        
        # Initialize speech recognition with optimized settings
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Ultra-fast recognition settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False  # Disable for speed
        self.recognizer.pause_threshold = 0.3  # Very short pause detection
        
        # Quick microphone calibration
        print("Quick microphone calibration...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
        
    def listen_for_speech(self, timeout=2):
        """Ultra-fast speech recognition"""
        try:
            with self.microphone as source:
                print(f"\n🎤 Listening... (speak within {timeout} seconds)")
                # Very short timeout and phrase limit
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=6)
            
            print("🔄 Processing...")
            # Fast recognition
            text = self.recognizer.recognize_google(audio)
            print(f"👤 You: {text}")
            return text
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate ultra-short responses for speed"""
        user_input_lower = user_input.lower()
        
        # Ultra-short responses
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hi! How are you?"
        
        elif any(question in user_input_lower for question in ['how are you']):
            return "I'm good! You?"
        
        elif any(word in user_input_lower for word in ['thank', 'thanks']):
            return "You're welcome!"
        
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'quit', 'exit']):
            return "Bye! See you later!"
        
        elif any(word in user_input_lower for word in ['time']):
            current_time = time.strftime("%I:%M")
            return f"It's {current_time}."
        
        else:
            # Ultra-short responses
            responses = [
                "Interesting!",
                "Tell me more.",
                "I see.",
                "Thanks!",
                "Go on.",
                "Okay.",
                "Right."
            ]
            import random
            return random.choice(responses)
    
    def synthesize_response(self, response_text):
        """Ultra-fast speech synthesis"""
        try:
            print(f"🤖 AI: {response_text}")
            print("🎵 Speaking...")
            
            # Generate speech
            wav = self.tts.infer(response_text, self.ref_codes, self.ref_text)
            
            # Quick audio normalization
            if np.max(np.abs(wav)) > 0:
                wav = wav / np.max(np.abs(wav)) * 0.7
            
            # Save and play audio
            output_path = os.path.abspath("temp_response.wav")
            sf.write(output_path, wav, 24000)
            
            # Play audio
            import subprocess
            subprocess.run(["afplay", output_path], check=False)
            
            # Quick cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
            
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    def run_conversation(self):
        """Ultra-fast conversation loop"""
        print("\n" + "="*60)
        print("⚡ ULTRA FAST VOICE CHAT")
        print("="*60)
        print("Speak quickly for instant responses!")
        print("Say 'bye' or 'quit' to exit.")
        print("="*60 + "\n")
        
        while True:
            # Ultra-fast listening
            user_input = self.listen_for_speech(timeout=2)
            
            if user_input is None:
                continue
            
            # Check for exit
            if any(word in user_input.lower() for word in ['goodbye', 'bye', 'quit', 'exit']):
                self.synthesize_response("Bye! See you later!")
                break
            
            # Generate and speak response
            response = self.generate_response(user_input)
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Ultra Fast Voice Chat with NeuTTS Air")
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
        # Initialize ultra fast voice chat
        chat = UltraFastVoiceChat(args.ref_audio, args.ref_text, args.backbone)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended. Bye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
