# Crowd Density Analysis System

A modern system to accurately evaluate and analyze crowd density in images for security awareness.

## Project Restructure

- `/backend/`: Python Flask service serving the `MC_CNN` PyTorch model.
- `/frontend/`: Modern vanilla Typescript application using Vite.
- `/Dockerfile`: Dockerized microservice config for the backend.
- (Note: The files have been restructured. You can safely remove the legacy UI_Service and ML_Service folders, as they are fully migrated.)

## How to Run

### 1. Backend Service (Dockerized)
The backend runs as a containerized microservice exposing the `/predict` API.

```bash
# Build the Docker image
docker build -t crowd-backend .

# Run the backend
docker run -p 8000:8000 crowd-backend
```

### 2. Frontend Application (Typescript + Vite)
The front end provides a modern UI with glassmorphism to interact with the backend service.

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:3000` to interact with the app. Upload images and observe the predicted density heatmaps and security recommendations dynamically rendered in the interface.
