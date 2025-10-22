# 🎙️ Edge Voice Chat - Complete Local AI Voice Assistant

A fully offline, privacy-focused voice chat system that runs entirely on your device using **Whisper** for speech recognition and **NeuTTS Air** for voice synthesis with voice cloning capabilities.

## 🌟 Features

- 🎤 **100% Offline Speech Recognition** - Powered by OpenAI Whisper
- 🎵 **Voice Cloning & Synthesis** - Powered by NeuTTS Air
- 🔒 **Complete Privacy** - Your data never leaves your device
- ⚡ **Real-time Processing** - Low latency voice conversations
- 🎯 **Multiple Interaction Modes** - Continuous listening, push-to-talk, text-to-voice
- 🧠 **Smart Conversations** - Context-aware, natural responses
- 📱 **Edge Computing** - Runs on laptops, mobile devices, Raspberry Pi
- 🌍 **Multi-language Support** - Whisper supports 99+ languages

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **macOS/Linux/Windows**
- **Microphone access**
- **~2GB storage** for models
- **4GB+ RAM** recommended

### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd edge-voice-chat
```

2. **Install dependencies:**

```bash
# Core dependencies
pip install -r requirements.txt

# Audio dependencies
brew install espeak portaudio  # macOS
# or
sudo apt install espeak portaudio19-dev  # Ubuntu

# Speech recognition
pip install openai-whisper SpeechRecognition pyaudio

# Optional: Push-to-talk
pip install keyboard
```

3. **Run the voice chat:**

```bash
# Basic voice chat
python3 examples/edge_voice_chat.py

# Push-to-talk (recommended for no feedback)
python3 examples/push_to_talk_chat.py

# Text-to-voice (no speech recognition needed)
python3 examples/text_to_voice_chat.py
```

## 📁 Project Structure

```
edge-voice-chat/
├── examples/
│   ├── edge_voice_chat.py          # Main edge voice chat
│   ├── push_to_talk_chat.py        # Push-to-talk version
│   ├── text_to_voice_chat.py       # Text input, voice output
│   ├── smart_edge_voice_chat.py    # Enhanced conversations
│   ├── fixed_edge_voice_chat.py    # Bug fixes version
│   ├── fast_voice_chat.py          # Optimized for speed
│   ├── ultra_fast_voice_chat.py    # Maximum speed
│   └── simple_voice_demo.py        # Demo with pre-written responses
├── samples/
│   ├── dave.wav                    # Reference voice sample
│   ├── dave.txt                    # Reference text
│   ├── jo.wav                      # Alternative voice sample
│   └── jo.txt                      # Alternative text
├── neuttsair/
│   └── neutts.py                   # NeuTTS Air implementation
├── requirements.txt                 # Python dependencies
├── install_whisper.sh              # Whisper installation script
└── README.md                       # This file
```

## 🎯 Usage Modes

### 1. Continuous Voice Chat

```bash
python3 examples/edge_voice_chat.py --whisper_model base
```

- **Best for**: Natural conversations
- **Requires**: Headphones to prevent feedback
- **Features**: Continuous listening, natural flow

### 2. Push-to-Talk (Recommended)

```bash
python3 examples/push_to_talk_chat.py --whisper_model base
```

- **Best for**: No audio feedback issues
- **Controls**: Hold SPACE to talk, release to stop
- **Features**: Complete control, professional quality

### 3. Text-to-Voice Chat

```bash
python3 examples/text_to_voice_chat.py
```

- **Best for**: Testing voice quality
- **Input**: Type messages, get voice responses
- **Features**: No speech recognition needed

### 4. Smart Conversations

```bash
python3 examples/smart_edge_voice_chat.py --whisper_model base
```

- **Best for**: Engaging conversations
- **Features**: Context awareness, memory, varied responses

## ⚙️ Configuration Options

### Whisper Models

```bash
--whisper_model tiny    # 39MB  - Fastest, basic accuracy
--whisper_model base    # 139MB - Good balance (recommended)
--whisper_model small   # 461MB - Better accuracy
--whisper_model medium  # 1.5GB - High accuracy
--whisper_model large   # 2.9GB - Best accuracy
```

### Voice Cloning

```bash
# Use Dave's voice (default)
python3 examples/edge_voice_chat.py --ref_audio samples/dave.wav --ref_text samples/dave.txt

# Use Jo's voice
python3 examples/edge_voice_chat.py --ref_audio samples/jo.wav --ref_text samples/jo.txt

# Use your own voice
python3 examples/edge_voice_chat.py --ref_audio your_voice.wav --ref_text your_text.txt
```

### Model Backends

```bash
# GGUF format (recommended for performance)
--backbone neuphonic/neutts-air-q4-gguf

# PyTorch format (better compatibility)
--backbone neuphonic/neutts-air
```

## 🎤 Audio Setup

### Preventing Audio Feedback

**Problem**: AI hears its own voice through microphone, creating echo/dual voices.

**Solutions**:

1. **Use headphones** (easiest)
2. **Push-to-talk mode** (best control)
3. **Lower speaker volume** (partial solution)

### Microphone Calibration

The system automatically calibrates your microphone on startup. For best results:

- Speak at normal volume
- Keep background noise minimal
- Use a good quality microphone
- Ensure proper microphone permissions

## 🔧 Troubleshooting

### Common Issues

**1. "espeak not installed" error:**

```bash
# macOS
brew install espeak

# Ubuntu/Debian
sudo apt install espeak

# Then apply macOS fix if needed (already included in code)
```

**2. Audio feedback/dual voices:**

- Use headphones instead of speakers
- Use push-to-talk mode: `python3 examples/push_to_talk_chat.py`
- Lower speaker volume

**3. Poor speech recognition:**

- Speak clearly and at normal volume
- Reduce background noise
- Try different Whisper models
- Check microphone permissions

**4. Slow performance:**

- Use smaller Whisper model: `--whisper_model tiny`
- Use GGUF backbone: `--backbone neuphonic/neutts-air-q4-gguf`
- Close other applications

**5. PyAudio installation issues:**

```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu
sudo apt install portaudio19-dev python3-pyaudio
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│  Whisper (SR)   │───▶│   AI Logic      │
│                 │    │   (Offline)     │    │  (Local)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Speakers      │◀───│ NeuTTS Air (TTS)│◀───│ Response Gen    │
│                 │    │   (Offline)     │    │  (Local)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Speech Input** → Microphone captures audio
2. **Speech Recognition** → Whisper converts speech to text
3. **Response Generation** → Local AI logic generates response
4. **Voice Synthesis** → NeuTTS Air converts text to speech
5. **Audio Output** → Speakers play the response

### Privacy & Security

- ✅ **No internet required** after initial setup
- ✅ **No cloud processing** - everything runs locally
- ✅ **No data collection** - your voice stays on your device
- ✅ **No tracking** - completely anonymous
- ✅ **Open source** - transparent and auditable

## 📊 Performance

### Hardware Requirements

| Component | Minimum          | Recommended        |
| --------- | ---------------- | ------------------ |
| RAM       | 2GB              | 8GB+               |
| Storage   | 2GB              | 5GB+               |
| CPU       | Any x86_64/ARM64 | Modern multi-core  |
| GPU       | Not required     | Optional for speed |

### Model Sizes

| Model          | Size  | Speed   | Accuracy | Use Case         |
| -------------- | ----- | ------- | -------- | ---------------- |
| Whisper Tiny   | 39MB  | Fastest | Basic    | Quick testing    |
| Whisper Base   | 139MB | Fast    | Good     | **Recommended**  |
| Whisper Small  | 461MB | Medium  | Better   | High quality     |
| Whisper Medium | 1.5GB | Slow    | High     | Maximum accuracy |
| Whisper Large  | 2.9GB | Slowest | Best     | Professional use |

### Performance Tips

1. **Use GGUF models** for better CPU performance
2. **Pre-encode references** to reduce latency
3. **Use smaller Whisper models** for faster recognition
4. **Close background apps** to free up resources
5. **Use headphones** to prevent feedback delays

## 🔮 Advanced Usage

### Creating Custom Voice Clones

1. **Record reference audio:**

   - 3-15 seconds of clean speech
   - Mono, 16-44kHz, WAV format
   - Minimal background noise
   - Natural, continuous speech

2. **Create reference text:**

   - Exact transcript of the audio
   - Plain text file
   - Same language as audio

3. **Use your custom voice:**

```bash
python3 examples/edge_voice_chat.py \
  --ref_audio my_voice.wav \
  --ref_text my_voice.txt
```

### Integration Examples

**Simple Python integration:**

```python
from neuttsair.neutts import NeuTTSAir
import whisper

# Initialize models
tts = NeuTTSAir(backbone_repo="neuphonic/neutts-air-q4-gguf")
whisper_model = whisper.load_model("base")

# Load reference
ref_codes = tts.encode_reference("samples/dave.wav")
with open("samples/dave.txt", "r") as f:
    ref_text = f.read().strip()

# Process speech
result = whisper_model.transcribe("input.wav")
response = generate_response(result["text"])
wav = tts.infer(response, ref_codes, ref_text)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd edge-voice-chat
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black examples/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Core Technologies

- **OpenAI Whisper** - Speech recognition engine that powers our offline voice understanding
- **Neuphonic NeuTTS Air** - Revolutionary voice synthesis and cloning technology
- **SpeechRecognition** - Python speech recognition library
- **PyAudio** - Audio I/O library for microphone and speaker control

### Original NeuTTS Air Contributors

Special thanks to the talented developers who created the amazing NeuTTS Air foundation:

- **Jiameng Gao** ([@jiamenguk](https://github.com/jiamenguk)) - Lead developer and creator of NeuTTS Air
- **Harry Julian** ([@harryjulian](https://github.com/harryjulian)) - Core contributor and developer
- **Tahir** ([@Tahirc1](https://github.com/Tahirc1)) - Contributor
- **Stan Stanislaus** - Contributor
- **Shagun Bera** ([@notV3NOM](https://github.com/notV3NOM)) - Contributor
- **Danil** ([@keepfocused](https://github.com/keepfocused)) - Contributor
- **Ali Tavallaie** - Contributor
- **Abubakar Abid** ([@islamrealm](https://github.com/islamrealm)) - Contributor

### Voice-Local-AI Integration

- **SebDigital** - Voice chat implementation, Whisper integration, edge computing features, and comprehensive documentation

This project builds upon the excellent work of the NeuTTS Air team, extending it with complete offline speech recognition and creating a privacy-first voice assistant experience.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

## 🎯 Roadmap

- [ ] Real-time streaming support
- [ ] Multi-speaker conversations
- [ ] Voice emotion detection
- [ ] Mobile app versions
- [ ] Web interface
- [ ] Plugin system
- [ ] Voice training interface

---

**🎉 Enjoy your completely private, offline voice AI assistant!**

_Built with ❤️ for privacy, performance, and open-source principles._
