import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Package, Grid3x3, Camera, Activity, TrendingUp, Clock, Box } from 'lucide-react';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Camera className="w-6 h-6" />,
      title: "Real-Time Object Tracking",
      description: "Track pallets across 11 warehouse cameras with advanced computer vision"
    },
    {
      icon: <Grid3x3 className="w-6 h-6" />,
      title: "Digital Twin Visualization",
      description: "Interactive 2D warehouse map with live object positioning"
    },
    {
      icon: <Activity className="w-6 h-6" />,
      title: "Cross-Camera Tracking",
      description: "Persistent object IDs maintained across multiple camera zones"
    },
    {
      icon: <TrendingUp className="w-6 h-6" />,
      title: "Performance Optimized",
      description: "GPU-accelerated processing with Redis-backed persistence"
    },
    {
      icon: <Clock className="w-6 h-6" />,
      title: "Live Monitoring",
      description: "Real-time detection updates with MongoDB Atlas integration"
    },
    {
      icon: <Box className="w-6 h-6" />,
      title: "Inventory Management",
      description: "QR code linking and automated data collection capabilities"
    }
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-center">
          <div className="flex items-center space-x-4">
            <img src="/logo3.png" alt="WARP Logo" className="w-24 h-18" />
            <h1 className="text-2xl md:text-2xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
              Digital Twin
            </h1>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-purple-900/20"></div>
        <div className="relative max-w-7xl mx-auto px-6 py-24">
          <div className="text-center">
            <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
              Warehouse Intelligence Platform
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Advanced computer vision system for real-time warehouse object tracking and digital twin visualization. 
              Monitor your entire warehouse with AI-powered detection across multiple camera zones.
            </p>
            
            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <button
                onClick={() => navigate('/login')}
                className="group relative px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold text-lg transition-all duration-300 hover:from-blue-500 hover:to-purple-500 hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-lg opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                <span className="relative">Admin Login</span>
              </button>
              
              <button
                onClick={() => navigate('/demo')}
                className="px-8 py-4 border-2 border-gray-600 rounded-lg font-semibold text-lg transition-all duration-300 hover:border-blue-500 hover:bg-blue-500/10 hover:scale-105"
              >
                Guest Mode (Demo)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-12">
          <h3 className="text-3xl font-bold mb-4">Platform Capabilities</h3>
          <p className="text-gray-400 text-lg">Powered by advanced AI and computer vision technology</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-gray-900 border border-gray-800 rounded-lg p-6 hover:border-blue-500/50 transition-all duration-300 hover:bg-gray-800/50"
            >
              <div className="flex items-center mb-4">
                <div className="p-2 bg-blue-600/20 rounded-lg text-blue-400 mr-4">
                  {feature.icon}
                </div>
                <h4 className="text-xl font-semibold">{feature.title}</h4>
              </div>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Technical Specs */}
      <div className="bg-gray-900 border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold mb-4">System Specifications</h3>
            <p className="text-gray-400 text-lg">Enterprise-grade performance and reliability</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400 mb-2">11</div>
              <div className="text-gray-400">Camera Zones</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">≥5 FPS</div>
              <div className="text-gray-400">Processing Speed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400 mb-2">1080p</div>
              <div className="text-gray-400">Video Resolution</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-400 mb-2">24/7</div>
              <div className="text-gray-400">Monitoring</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-950 border-t border-gray-800 px-6 py-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-4 mb-4">
            <img src="/logo3.png" alt="WARP Logo" className="w-12 h-9" />
            <span className="text-lg font-semibold text-gray-400">WARP Digital Twin Platform</span>
          </div>
          <p className="text-gray-500 text-sm">
            Advanced warehouse intelligence powered by computer vision and AI
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
