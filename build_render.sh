#!/bin/bash

# This script helps prepare the project for Render deployment

echo "Starting Render deployment preparation..."

# Install dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

# Create any necessary directories
mkdir -p backend/logs

# Check if pyttsx3 is causing issues (common on cloud platforms)
if pip show pyttsx3 > /dev/null; then
  echo "Note: pyttsx3 is installed but might not work in cloud environments."
  echo "Text-to-speech functionality may be limited on Render."
fi

echo "Render deployment preparation complete!" 