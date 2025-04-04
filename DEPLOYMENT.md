# Deployment Guide for ElderlyCareUI

This guide describes how to deploy the ElderlyCareUI project to free hosting platforms. The project consists of two main parts:
1. Backend (Flask API) - will be deployed to Render.com
2. Frontend (React + Vite) - will be deployed to Vercel.com

## Prerequisites

- A GitHub account
- Git installed on your local machine
- A Render.com account (free tier)
- A Vercel.com account (free tier)

## Step 1: Push Your Code to GitHub

1. Create a new repository on GitHub
2. Initialize Git in your project (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/ElderlyCareUI.git
   git push -u origin main
   ```

## Step 2: Deploy the Backend to Render.com

### Option 1: Deploy using Render Dashboard UI

1. Sign up/login to [Render.com](https://render.com)
2. Click "New Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `elderlycareui-backend` (or your preferred name)
   - **Environment**: Python
   - **Region**: Choose the closest to your target users
   - **Branch**: main
   - **Build Command**: `pip install -r backend/requirements.txt && chmod +x build_render.sh && ./build_render.sh`
   - **Start Command**: `cd backend && gunicorn app:app`
   - **Plan**: Free

5. Click "Create Web Service"
6. Wait for the deployment to complete (this may take a few minutes)
7. Once deployed, note the URL (typically `https://your-service-name.onrender.com`)

### Option 2: Deploy using render.yaml (Blueprint)

1. Sign up/login to [Render.com](https://render.com)
2. Go to "Blueprints" section 
3. Click "New Blueprint Instance"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file and create the services defined in it
6. Review the configuration and click "Apply"
7. Wait for the deployment to complete
8. Once deployed, note the URL for your backend service

### Troubleshooting Render Deployment

If you encounter issues with the deployment on Render:

1. **Build Failures**:
   - Check the build logs in the Render dashboard
   - Make sure your Python dependencies are compatible with the Python version
   - The `build_render.sh` script should help handle pyttsx3 issues

2. **Runtime Errors**:
   - Check if the application is starting by looking at the logs
   - Verify environment variables are set correctly
   - Check the Health Check is passing

3. **Voice Reminder Service Issues**:
   - The application automatically detects cloud environments and disables pyttsx3
   - Text-to-speech functionality will be limited, but APIs will continue to work

4. **Database Issues**:
   - The default SQLite database may reset on Render free tier when the instance sleeps
   - Consider upgrading to a persistent database service for production

## Step 3: Update Frontend Configuration

After deploying the backend, you need to update the frontend configuration:

1. Edit `vercel.json` to update the backend URL:
   ```json
   {
     "routes": [
       { "handle": "filesystem" },
       { "src": "/api/(.*)", "dest": "https://your-backend-url.onrender.com/api/$1" },
       { "src": "/(.*)", "dest": "index.html" }
     ],
     "env": {
       "VITE_API_BASE_URL": "https://your-backend-url.onrender.com"
     }
   }
   ```

2. Commit and push these changes:
   ```bash
   git add vercel.json
   git commit -m "Update backend URL"
   git push
   ```

## Step 4: Deploy the Frontend to Vercel.com

1. Sign up/login to [Vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build` (should be detected automatically)
   - **Output Directory**: `dist` (should be detected automatically)

5. Click "Deploy"
6. Wait for the deployment to complete
7. Once deployed, Vercel will provide you with a URL for your application

## Step 5: Testing and Verification

1. Visit your frontend URL to ensure it's working correctly
2. Test API endpoints through the frontend
3. If there are issues:
   - Check Render logs for backend problems
   - Check Vercel logs for frontend problems

## Keeping Your Application Running

The free tier of Render will spin down your service after 15 minutes of inactivity:

1. **Avoid Spin Down**: Set up a free uptime monitoring service like UptimeRobot to ping your backend URL every 5-10 minutes
2. **Restart on Error**: Render automatically restarts your service if it crashes

## Updating Your Application

When you make changes to your code:

1. Commit and push changes to GitHub
2. Render and Vercel will automatically redeploy your application

## Environment Variables

If your application requires environment variables:

- For backend: Add them in the Render dashboard under "Environment" section
- For frontend: Add them in the Vercel dashboard under "Environment Variables" section

## Custom Domains (Optional)

Both Render and Vercel allow you to set up custom domains:

- For Render: Go to your service > Settings > Custom Domain
- For Vercel: Go to your project > Settings > Domains

Note that SSL certificates are provided automatically by both services. 