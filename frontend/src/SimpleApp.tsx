import React, { useState } from 'react';
import WorkingWarehouseView from './components/WorkingWarehouseView';

const SimpleApp: React.FC = () => {
  const [viewMode, setViewMode] = useState<'tracking' | 'grid'>('tracking');

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Modern Header */}
      <header className="bg-gray-900 border-b border-gray-700 p-6 shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Digital Twin
            </h1>
            <p className="text-gray-400 text-sm mt-1">Real-time Warehouse Object Tracking</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-gray-800 px-3 py-2 rounded-lg border border-gray-600">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-300">System Online</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="p-8">
        {/* Modern View Toggle */}
        <div className="mb-8">
          <div className="flex bg-gray-800 rounded-xl p-1.5 w-fit border border-gray-600 shadow-lg">
            <button
              onClick={() => setViewMode('tracking')}
              className={`px-6 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                viewMode === 'tracking'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-gray-700'
              }`}
            >
              🎯 Live Tracking
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`px-6 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                viewMode === 'grid'
                  ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-gray-700'
              }`}
            >
              📊 Grid View
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div style={{ height: '600px' }}>
          {viewMode === 'tracking' ? (
            <WorkingWarehouseView />
          ) : (
            <div className="bg-gray-900 rounded-xl p-8 h-full flex items-center justify-center border border-gray-700 shadow-2xl">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                  <span className="text-2xl">📊</span>
                </div>
                <h2 className="text-2xl font-bold mb-4 text-white">Grid View</h2>
                <div className="text-gray-400 max-w-md">
                  Traditional grid-based warehouse layout view will be available here.
                  Switch to Live Tracking to see real-time object detection.
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modern Stats Cards */}
        <div className="mt-8 grid grid-cols-3 gap-6">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="w-10 h-10 bg-blue-500 bg-opacity-20 rounded-lg flex items-center justify-center">
                <span className="text-blue-400 text-lg">🎯</span>
              </div>
              <div className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded-full">Live</div>
            </div>
            <div className="text-3xl font-bold text-blue-400 mb-1">8</div>
            <div className="text-sm text-gray-400">Tracked Objects</div>
          </div>
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="w-10 h-10 bg-green-500 bg-opacity-20 rounded-lg flex items-center justify-center">
                <span className="text-green-400 text-lg">📈</span>
              </div>
              <div className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded-full">Total</div>
            </div>
            <div className="text-3xl font-bold text-green-400 mb-1">156</div>
            <div className="text-sm text-gray-400">Total Detections</div>
          </div>
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-700 shadow-lg hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="w-10 h-10 bg-yellow-500 bg-opacity-20 rounded-lg flex items-center justify-center">
                <span className="text-yellow-400 text-lg">⚡</span>
              </div>
              <div className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded-full">Recent</div>
            </div>
            <div className="text-3xl font-bold text-yellow-400 mb-1">3</div>
            <div className="text-sm text-gray-400">Recent Activity</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleApp;
