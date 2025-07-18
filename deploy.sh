#!/bin/bash

# 🚀 MINIMAL DEPLOYMENT SCRIPT - NO CODE CHANGES REQUIRED
# This script deploys your warehouse tracking system with zero code modifications

echo "🚀 DEPLOYING WAREHOUSE TRACKING SYSTEM"
echo "======================================"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Please ensure your .env file is in the project root"
    exit 1
fi

echo "✅ .env file found"

# Option 1: Docker Deployment (Recommended)
echo ""
echo "DEPLOYMENT OPTIONS:"
echo "1. Docker Deployment (Recommended - One command)"
echo "2. Manual Deployment (Traditional server setup)"
echo ""
read -p "Choose deployment method (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "🐳 DOCKER DEPLOYMENT"
    echo "==================="
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        echo "✅ Docker installed. Please log out and back in, then run this script again."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose not found. Installing..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    echo "🏗️  Building and starting containers..."
    docker-compose up --build -d
    
    echo ""
    echo "✅ DEPLOYMENT COMPLETE!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📊 API Docs: http://localhost:8000/docs"
    echo ""
    echo "📋 MANAGEMENT COMMANDS:"
    echo "   View logs: docker-compose logs -f"
    echo "   Stop: docker-compose down"
    echo "   Restart: docker-compose restart"

elif [ "$choice" = "2" ]; then
    echo ""
    echo "🔧 MANUAL DEPLOYMENT"
    echo "==================="
    
    # Install system dependencies
    echo "📦 Installing system dependencies..."
    sudo apt update
    sudo apt install -y nodejs npm python3 python3-pip nginx curl
    
    # Install Python dependencies
    echo "🐍 Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    # Build frontend
    echo "⚛️  Building frontend..."
    cd frontend
    npm install
    npm run build
    cd ..
    
    # Create systemd service for backend
    echo "🔧 Creating backend service..."
    sudo tee /etc/systemd/system/warehouse-backend.service > /dev/null <<EOF
[Unit]
Description=Warehouse Tracking Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=/usr/bin:/usr/local/bin
Environment=PYTHONPATH=$(pwd)
ExecStart=/usr/bin/python3 backend/live_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    # Create systemd service for frontend
    echo "🌐 Creating frontend service..."
    sudo tee /etc/systemd/system/warehouse-frontend.service > /dev/null <<EOF
[Unit]
Description=Warehouse Tracking Frontend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)/frontend
ExecStart=/usr/bin/npx serve -s dist -l 3000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    # Start services
    echo "🚀 Starting services..."
    sudo systemctl daemon-reload
    sudo systemctl enable warehouse-backend warehouse-frontend
    sudo systemctl start warehouse-backend warehouse-frontend
    
    echo ""
    echo "✅ DEPLOYMENT COMPLETE!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo ""
    echo "📋 MANAGEMENT COMMANDS:"
    echo "   Check status: sudo systemctl status warehouse-backend warehouse-frontend"
    echo "   View logs: sudo journalctl -u warehouse-backend -f"
    echo "   Restart: sudo systemctl restart warehouse-backend warehouse-frontend"

else
    echo "❌ Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "Your warehouse tracking system is now running!"
