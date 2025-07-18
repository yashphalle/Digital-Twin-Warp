# Backend-only Dockerfile (no CV dependencies)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files for backend
COPY backend/ ./backend/
COPY cv/configs/ ./cv/configs/

# Expose port
EXPOSE 8000

# Run the backend server
CMD ["python", "backend/live_server.py"]
