#!/usr/bin/env python3
"""
Smart Edge Voice Chat - Enhanced conversation with better responses
Uses Whisper for offline speech recognition + NeuTTS Air for voice synthesis
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


class SmartEdgeVoiceChat:
    def __init__(self, ref_audio_path, ref_text_path, backbone="neuphonic/neutts-air-q4-gguf", whisper_model="base"):
        """Initialize the smart edge voice chat system"""
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
        
        # Conversation context and memory
        self.conversation_history = []
        self.user_name = None
        self.topics_discussed = set()
        self.response_count = 0
        
        print("🚀 Smart Edge Voice Chat Ready!")
        
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
    
    def extract_keywords(self, text):
        """Extract important keywords from user input"""
        keywords = []
        text_lower = text.lower()
        
        # Common topics
        topics = {
            'work': ['work', 'job', 'career', 'office', 'business', 'company'],
            'family': ['family', 'parents', 'mother', 'father', 'sister', 'brother', 'children', 'kids'],
            'hobbies': ['hobby', 'hobbies', 'sport', 'music', 'art', 'reading', 'gaming', 'travel'],
            'food': ['food', 'eat', 'restaurant', 'cooking', 'recipe', 'meal', 'hungry'],
            'weather': ['weather', 'rain', 'sunny', 'cloudy', 'hot', 'cold', 'temperature'],
            'technology': ['computer', 'phone', 'internet', 'ai', 'technology', 'software', 'app'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'tired', 'worried', 'nervous', 'calm'],
            'time': ['time', 'clock', 'hour', 'minute', 'today', 'tomorrow', 'yesterday', 'weekend'],
            'location': ['home', 'work', 'school', 'store', 'park', 'city', 'country', 'travel']
        }
        
        for topic, words in topics.items():
            if any(word in text_lower for word in words):
                keywords.append(topic)
        
        return keywords
    
    def generate_smart_response(self, user_input):
        """Generate intelligent, contextual responses"""
        user_input_lower = user_input.lower()
        self.response_count += 1
        
        # Extract keywords for context
        keywords = self.extract_keywords(user_input)
        self.topics_discussed.update(keywords)
        
        # Store in conversation history
        self.conversation_history.append(("user", user_input))
        
        # Extract name if mentioned
        if not self.user_name and any(word in user_input_lower for word in ['my name is', 'i am', 'i\'m']):
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ['name', 'is', 'am', 'i\'m'] and i + 1 < len(words):
                    self.user_name = words[i + 1].strip('.,!?')
                    break
        
        # Personalized greetings
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            if self.user_name:
                responses = [
                    f"Hello {self.user_name}! Great to see you again. How are you doing today?",
                    f"Hi there {self.user_name}! I'm doing well, thanks for asking. What's on your mind?",
                    f"Hey {self.user_name}! Good to chat with you. How has your day been so far?"
                ]
            else:
                responses = [
                    "Hello! Nice to meet you. I'm an AI assistant running completely offline on your device. What's your name?",
                    "Hi there! Great to meet you. I'm processing everything locally for your privacy. What should I call you?",
                    "Hey! I'm excited to chat with you. Everything we discuss stays on your device. What's your name?"
                ]
            return random.choice(responses)
        
        # How are you responses
        elif any(question in user_input_lower for question in ['how are you', 'how do you do', 'how\'s it going', 'what\'s up']):
            responses = [
                "I'm doing great! I love running completely offline on your device. How about you?",
                "I'm fantastic! It's so cool that we can chat with complete privacy. What's going on with you?",
                "I'm wonderful! Being able to process everything locally means your data stays safe. How are you feeling today?",
                "I'm excellent! This edge computing setup is really impressive. What's new with you?",
                "I'm doing really well! I appreciate that we can have this conversation without any cloud processing. How's your day going?"
            ]
            return random.choice(responses)
        
        # Name-related responses
        elif any(question in user_input_lower for question in ['what is your name', 'who are you', 'what\'s your name']):
            responses = [
                "I'm an AI assistant powered by Whisper for speech recognition and NeuTTS Air for voice synthesis. Everything runs locally on your device!",
                "I'm your local AI companion! I use Whisper to understand your speech and NeuTTS Air to respond with this cloned voice, all running offline.",
                "I'm an edge AI assistant! I'm designed to run completely on your device using Whisper and NeuTTS Air for privacy and speed.",
                "I'm your private AI assistant! I use cutting-edge local speech recognition and voice cloning technology, all running on your laptop."
            ]
            return random.choice(responses)
        
        # Time responses
        elif any(word in user_input_lower for word in ['time', 'clock', 'hour']):
            current_time = time.strftime("%I:%M %p")
            day_name = time.strftime("%A")
            responses = [
                f"The current time is {current_time} on {day_name}. How's your day going?",
                f"It's {current_time} right now. What do you have planned for today?",
                f"The time is {current_time}. Is there anything special you'd like to do today?",
                f"It's currently {current_time}. How has your {day_name} been treating you?"
            ]
            return random.choice(responses)
        
        # Privacy/offline responses
        elif any(word in user_input_lower for word in ['offline', 'local', 'edge', 'privacy', 'private']):
            responses = [
                "Yes! I'm running completely offline. Your voice data never leaves your device, ensuring complete privacy and security.",
                "Absolutely! Everything is processed locally on your device. No cloud, no internet required after setup - just pure privacy.",
                "That's right! I'm an edge AI assistant. All speech recognition and voice synthesis happens on your laptop for maximum privacy.",
                "Exactly! I'm designed for edge computing. Your conversations stay private and secure on your device."
            ]
            return random.choice(responses)
        
        # Technology responses
        elif 'technology' in keywords or any(word in user_input_lower for word in ['whisper', 'speech recognition', 'neutts', 'ai']):
            responses = [
                "I'm using Whisper for speech recognition and NeuTTS Air for voice synthesis, both running locally on your device!",
                "The technology is amazing! Whisper handles speech-to-text and NeuTTS Air creates this natural voice, all offline.",
                "It's pretty cool! Whisper understands your speech and NeuTTS Air generates my responses in this cloned voice.",
                "I'm powered by cutting-edge local AI! Whisper for listening and NeuTTS Air for speaking, all on your device."
            ]
            return random.choice(responses)
        
        # Work-related responses
        elif 'work' in keywords:
            responses = [
                "Work can be both challenging and rewarding. What kind of work do you do?",
                "I find work discussions really interesting. What's your profession?",
                "Work is such an important part of life. Tell me about your job or career.",
                "I'd love to hear about your work! What field are you in?"
            ]
            return random.choice(responses)
        
        # Family responses
        elif 'family' in keywords:
            responses = [
                "Family is so important! Tell me about your family if you'd like to share.",
                "I love hearing about families! What's your family like?",
                "Family relationships are so meaningful. I'd be happy to listen if you want to talk about yours.",
                "Family is wonderful! Feel free to share whatever you'd like about your family."
            ]
            return random.choice(responses)
        
        # Hobby responses
        elif 'hobbies' in keywords:
            responses = [
                "Hobbies make life so much more interesting! What are your favorite hobbies?",
                "I love talking about hobbies! What do you enjoy doing in your free time?",
                "Hobbies are great for relaxation and fun. What activities do you love?",
                "Personal interests are fascinating! What hobbies do you have?"
            ]
            return random.choice(responses)
        
        # Food responses
        elif 'food' in keywords:
            responses = [
                "Food brings people together! What kind of cuisine do you enjoy?",
                "I love food discussions! What's your favorite type of food?",
                "Food is such a universal topic! What do you like to eat?",
                "Cooking and eating are such pleasures! What are your food preferences?"
            ]
            return random.choice(responses)
        
        # Emotional responses
        elif 'emotions' in keywords:
            if any(word in user_input_lower for word in ['happy', 'excited', 'joyful', 'great', 'wonderful']):
                responses = [
                    "That's fantastic! I'm so glad you're feeling positive. What's making you happy?",
                    "Wonderful! It's great to hear you're in a good mood. Tell me more!",
                    "That's wonderful news! I love hearing when people are happy. What's the good news?",
                    "Excellent! Positive emotions are so uplifting. What's going well for you?"
                ]
            elif any(word in user_input_lower for word in ['sad', 'upset', 'disappointed', 'worried']):
                responses = [
                    "I'm sorry to hear you're feeling that way. Sometimes it helps to talk about what's bothering you.",
                    "I understand that can be difficult. I'm here to listen if you want to share more.",
                    "That sounds challenging. Remember that it's okay to feel this way, and talking can help.",
                    "I'm sorry you're going through a tough time. Would you like to talk about what's on your mind?"
                ]
            else:
                responses = [
                    "Emotions are such an important part of being human. How are you feeling about everything?",
                    "Feelings can be complex sometimes. I'm here to listen if you want to share.",
                    "Emotional well-being is so important. How are you doing emotionally?"
                ]
            return random.choice(responses)
        
        # Question responses
        elif any(question in user_input_lower for question in ['what', 'who', 'where', 'when', 'why', 'how']):
            responses = [
                "That's an interesting question! I'm still learning, but I'd be happy to discuss it with you.",
                "Great question! While I'm running locally on your device, I'd love to explore that topic together.",
                "That's a thoughtful question! I'm processing everything offline, but let's talk about it.",
                "Interesting point! I'm designed for privacy and local processing, but I enjoy these conversations."
            ]
            return random.choice(responses)
        
        # Thank you responses
        elif any(word in user_input_lower for word in ['thank', 'thanks', 'thank you']):
            responses = [
                "You're very welcome! I'm glad I could help. Is there anything else you'd like to discuss?",
                "My pleasure! I love having these conversations while keeping everything private on your device.",
                "You're welcome! It's great that we can chat with complete privacy. What else is on your mind?",
                "Happy to help! I appreciate that we can have this conversation locally. Anything else you'd like to talk about?"
            ]
            return random.choice(responses)
        
        # Goodbye responses
        elif any(word in user_input_lower for word in ['goodbye', 'bye', 'see you', 'farewell', 'quit', 'exit']):
            if self.user_name:
                responses = [
                    f"Goodbye {self.user_name}! It was great talking with you. Thanks for using our private edge chat system!",
                    f"See you later {self.user_name}! I enjoyed our conversation. Everything stayed private on your device.",
                    f"Bye {self.user_name}! Thanks for the chat. Your privacy was completely protected throughout our conversation.",
                    f"Farewell {self.user_name}! It was wonderful talking with you using this local AI system."
                ]
            else:
                responses = [
                    "Goodbye! It was great talking with you. Thanks for using the edge voice chat system!",
                    "See you later! I enjoyed our conversation. Everything stayed private on your device.",
                    "Bye! Thanks for the chat. Your privacy was completely protected throughout our conversation.",
                    "Farewell! It was wonderful talking with you using this local AI system."
                ]
            return random.choice(responses)
        
        # Contextual responses based on conversation history
        elif len(self.conversation_history) > 2:
            recent_topics = list(self.topics_discussed)[-3:]  # Last 3 topics
            if recent_topics:
                responses = [
                    f"That's interesting! Building on what we were discussing about {', '.join(recent_topics)}, I'd love to hear more.",
                    f"Great point! This relates to our earlier conversation about {recent_topics[0] if recent_topics else 'things'}. Tell me more.",
                    f"I see what you mean! This connects to our discussion about {recent_topics[0] if recent_topics else 'topics'}. What else can you share?",
                    f"That's a good perspective! It reminds me of what we talked about regarding {recent_topics[0] if recent_topics else 'various topics'}. Continue!"
                ]
            else:
                responses = [
                    "That's really interesting! I'm learning a lot from our conversation. Please continue.",
                    "I find that fascinating! Our conversation is getting quite engaging. Tell me more.",
                    "That's a great point! I'm enjoying our chat. What else would you like to share?",
                    "Interesting perspective! I'm processing everything locally and learning from our discussion. Go on!"
                ]
            return random.choice(responses)
        
        # Default responses with variety
        else:
            # More varied default responses
            responses = [
                "That's really interesting! I'm processing this completely offline on your device. Tell me more about that.",
                "I understand what you're saying. All my responses are generated locally for your privacy. Can you elaborate?",
                "Thanks for sharing that with me. Your data stays on your device. What else is on your mind?",
                "That's fascinating! I'm running entirely offline. I'd love to hear more about your thoughts.",
                "I appreciate you telling me that. Everything is processed locally. How do you feel about it?",
                "That's a great point. All processing happens on your device for privacy. What made you think of that?",
                "I'm listening and learning, all while keeping your data private and local. Please continue.",
                "That's something I hadn't thought about. Can you tell me more? I'm processing everything offline.",
                "I find that really interesting. What's your perspective on this? Everything stays on your device.",
                "Thanks for bringing that up. It's definitely worth discussing, and our conversation stays private.",
                "That's a thoughtful comment. What else would you like to share? I'm here to listen.",
                "I appreciate your input. This is a great conversation, and it's all happening locally on your device.",
                "That's an interesting way to look at it. Tell me more about your thoughts on this.",
                "I'm enjoying our chat. What other topics interest you? Everything we discuss stays private.",
                "That's worth thinking about. What are your thoughts on this? I'm processing everything offline."
            ]
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
        """Main conversation loop with smart responses"""
        print("\n" + "="*70)
        print("🧠 SMART EDGE VOICE CHAT - ENHANCED CONVERSATIONS")
        print("="*70)
        if self.use_whisper:
            print("✅ Whisper Speech Recognition (offline)")
        else:
            print("⚠️ Google Speech Recognition (requires internet)")
        print("✅ NeuTTS Air Voice Synthesis (offline)")
        print("✅ Smart Context-Aware Responses")
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
                response = self.generate_smart_response(user_input)
                self.synthesize_response(response)
                break
            
            # Generate smart response
            response = self.generate_smart_response(user_input)
            
            # Synthesize and speak response
            self.synthesize_response(response)


def main():
    parser = argparse.ArgumentParser(description="Smart Edge Voice Chat - Enhanced Conversations")
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
        # Initialize smart edge voice chat
        chat = SmartEdgeVoiceChat(args.ref_audio, args.ref_text, args.backbone, args.whisper_model)
        
        # Start conversation
        chat.run_conversation()
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation ended by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
