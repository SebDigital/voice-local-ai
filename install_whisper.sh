#!/bin/bash

echo "🚀 Installing Whisper for Edge Speech Recognition..."
echo "=================================================="

# Install Whisper
echo "📦 Installing OpenAI Whisper..."
pip install openai-whisper

echo "✅ Whisper installation complete!"
echo ""
echo "🎯 Whisper Model Sizes:"
echo "   • tiny    - Fastest, least accurate (~1GB)"
echo "   • base    - Good balance (~1GB)"
echo "   • small   - Better accuracy (~2GB)"
echo "   • medium  - High accuracy (~5GB)"
echo "   • large   - Best accuracy (~10GB)"
echo ""
echo "🚀 To run edge voice chat:"
echo "   python3 examples/edge_voice_chat.py --whisper_model base"
echo ""
echo "🔒 100% Offline - No internet required after setup!"
