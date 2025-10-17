#!/usr/bin/env python3
"""
Text-to-Voice Chat - Type messages and get voice responses
No speech recognition needed - perfect for testing TTS quality
"""

import argparse
import time
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import neuttsair
sys.path.append(str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir


class TextToVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf"):
        """Initialize the text-to-voice chat system"""
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
        print("Ready for text-to-voice chat!")
        
    def generate_response(self, user_input):
        """Generate intelligent responses"""
        user_input_lower = user_input.lower()
        
        # Comprehensive response patterns
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
        
        elif any(word in user_input_lower for word in ['love', 'like', 'enjoy', 'favorite']):
            return "That's wonderful! It's great to hear about things you love. What else do you enjoy?"
        
        elif any(word in user_input_lower for word in ['hate', 'dislike', 'annoying']):
            return "I understand that can be frustrating. Would you like to talk about something more positive?"
        
        elif any(word in user_input_lower for word in ['work', 'job', 'career']):
            return "Work can be both challenging and rewarding. What do you do for a living?"
        
        elif any(word in user_input_lower for word in ['family', 'parents', 'siblings', 'children']):
            return "Family is so important. Tell me about your family if you'd like."
        
        elif any(word in user_input_lower for word in ['friend', 'friends', 'social']):
            return "Friends are wonderful to have. What do you like to do with your friends?"
        
        elif any(word in user_input_lower for word in ['music', 'song', 'band', 'artist']):
            return "Music is such a universal language. What kind of music do you enjoy?"
        
        elif any(word in user_input_lower for word in ['movie', 'film', 'cinema', 'actor']):
            return "Movies are great entertainment! What's your favorite type of movie?"
        
        elif any(word in user_input_lower for word in ['book', 'reading', 'novel', 'author']):
            return "Reading is such a wonderful hobby. What kind of books do you enjoy?"
        
        elif any(word in user_input_lower for word in ['food', 'eat', 'restaurant', 'cooking']):
            return "Food brings people together! What's your favorite type of cuisine?"
        
        elif any(word in user_input_lower for word in ['travel', 'vacation', 'trip', 'visit']):
            return "Traveling is so exciting! Where would you like to go or where have you been?"
        
        elif any(word in user_input_lower for word in ['sport', 'exercise', 'fitness', 'gym']):
            return "Staying active is important for health. What kind of sports or exercise do you enjoy?"
        
        elif any(word in user_input_lower for word in ['hobby', 'interest', 'passion']):
            return "Hobbies make life more interesting! What are your favorite hobbies?"
        
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
                "Thanks for bringing that up. It's definitely worth discussing.",
                "That's a thoughtful comment. What else would you like to share?",
                "I appreciate your input. This is a great conversation!",
                "That's an interesting way to look at it. Tell me more.",
                "I'm enjoying our chat. What other topics interest you?",
                "That's worth thinking about. What are your thoughts on this?"
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
        """Main conversation loop - type messages and get voice responses"""
        print("\n" + "="*70)
        print("⌨️  TEXT-TO-VOICE CHAT WITH NEUTTS AIR")
        print("="*70)
        print("💡 Type your messages and I'll respond with voice!")
        print("   • Type 'quit', 'exit', or 'goodbye' to end")
        print("   • Press Enter to send your message")
        print("   • The AI will respond with Dave's cloned voice")
        print("="*70 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if any(word in user_input.lower() for word in ['goodbye', 'quit', 'exit', 'stop', 'bye']):
                    self.synthesize_response("Goodbye! It was great talking with you. Have a wonderful day!")
                    break
                
                # Generate response
                response = self.generate_response(user_input)
                
                # Synthesize and speak response
                self.synthesize_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Conversation ended by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Conversation ended. Goodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="Text-to-Voice Chat with NeuTTS Air")
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
        # Initialize text-to-voice chat
        chat = TextToVoiceChat(args.ref_audio, args.ref_text, args.backbone)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
