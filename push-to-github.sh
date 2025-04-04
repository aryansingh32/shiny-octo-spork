#!/bin/bash

# This script pushes the latest changes to GitHub

echo "Pushing latest changes to GitHub..."

# Check if remote exists
if ! git remote | grep origin > /dev/null; then
  echo "No 'origin' remote found."
  echo "Please specify the GitHub repository URL:"
  read repo_url
  git remote add origin $repo_url
fi

# Push to GitHub
git push -u origin main

echo "Push completed!"
echo "Visit your Render dashboard to deploy the latest changes." 