#!/bin/bash

# This script is used by Vercel to install dependencies and build the project
echo "Running custom build script for Vercel deployment"

# Install dependencies with legacy peer deps to fix dependency conflicts
echo "Installing dependencies with legacy-peer-deps..."
npm install --legacy-peer-deps

# Run the build
echo "Building the project..."
npm run build

echo "Build completed successfully!" 