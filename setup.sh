#!/bin/bash

echo "🚀 Edge Voice Chat Setup Script"
echo "================================"
echo ""

# Check if Python 3.11+ is installed
echo "🐍 Checking Python version..."
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version+ required, found $python_version"
    echo "Please install Python 3.11 or later"
    exit 1
fi

# Detect OS
echo ""
echo "🖥️  Detecting operating system..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "✅ macOS detected"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "✅ Linux detected"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    echo "✅ Windows detected"
else
    OS="unknown"
    echo "⚠️  Unknown OS: $OSTYPE"
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
if [ "$OS" = "macos" ]; then
    echo "Installing dependencies for macOS..."
    if command -v brew &> /dev/null; then
        brew install espeak portaudio
        echo "✅ Homebrew packages installed"
    else
        echo "⚠️  Homebrew not found. Please install espeak and portaudio manually:"
        echo "   brew install espeak portaudio"
    fi
elif [ "$OS" = "linux" ]; then
    echo "Installing dependencies for Linux..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y espeak portaudio19-dev python3-pyaudio
        echo "✅ APT packages installed"
    elif command -v yum &> /dev/null; then
        sudo yum install -y espeak portaudio-devel
        echo "✅ YUM packages installed"
    else
        echo "⚠️  Package manager not found. Please install espeak and portaudio manually"
    fi
elif [ "$OS" = "windows" ]; then
    echo "⚠️  Windows detected. Please install dependencies manually:"
    echo "   1. Install espeak from: http://espeak.sourceforge.net/"
    echo "   2. Install portaudio from: http://www.portaudio.com/"
fi

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements_edge_voice_chat.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    echo "Try installing manually: pip3 install -r requirements_edge_voice_chat.txt"
    exit 1
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "
try:
    import whisper
    import speech_recognition
    import pyaudio
    import soundfile
    print('✅ Core dependencies working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Download Whisper model
echo ""
echo "📥 Downloading Whisper base model..."
python3 -c "
import whisper
print('Downloading Whisper base model (this may take a few minutes)...')
model = whisper.load_model('base')
print('✅ Whisper model downloaded and ready')
"

# Test NeuTTS Air
echo ""
echo "🧪 Testing NeuTTS Air..."
python3 -c "
try:
    from neuttsair.neutts import NeuTTSAir
    print('✅ NeuTTS Air import successful')
except ImportError as e:
    print(f'❌ NeuTTS Air import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🚀 Ready to use Edge Voice Chat:"
echo ""
echo "1. Basic voice chat:"
echo "   python3 examples/edge_voice_chat.py"
echo ""
echo "2. Push-to-talk (recommended):"
echo "   python3 examples/push_to_talk_chat.py"
echo ""
echo "3. Text-to-voice:"
echo "   python3 examples/text_to_voice_chat.py"
echo ""
echo "4. Smart conversations:"
echo "   python3 examples/smart_edge_voice_chat.py"
echo ""
echo "💡 Tips:"
echo "   • Use headphones to prevent audio feedback"
echo "   • Push-to-talk mode prevents echo issues"
echo "   • All processing happens locally on your device"
echo "   • Your voice data never leaves your device"
echo ""
echo "🔒 Complete Privacy - No Internet Required!"
echo ""
