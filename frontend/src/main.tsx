import React, { useState, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'

// Icons (using text for simplicity)
const Package = () => <span>📦</span>;
const Activity = () => <span>📊</span>;
const Clock = () => <span>⏰</span>;
const TrendingUp = () => <span>📈</span>;
const Camera = () => <span>📹</span>;
const Search = () => <span>🔍</span>;
const Bell = () => <span>🔔</span>;
const User = () => <span>👤</span>;

// Stats Card Component
const StatCard = ({ icon, label, value, trend, trendLabel, color }) => {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
    red: 'text-red-400'
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <div className={`${colorClasses[color]} text-lg`}>{icon}</div>
        {trend && (
          <div className="text-xs text-green-400">
            +{trend}% {trendLabel}
          </div>
        )}
      </div>
      <div className={`text-2xl font-bold ${colorClasses[color]} mb-1`}>{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  );
};

// Camera Feed Component
const CameraFeed = ({ cameraId, status }) => {
  const [streamError, setStreamError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const handleImageLoad = () => {
    setIsLoading(false);
    setStreamError(false);
  };

  const handleImageError = () => {
    setIsLoading(false);
    setStreamError(true);
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
      <div className="bg-gray-700 px-3 py-2 flex items-center justify-between">
        <span className="text-sm font-medium text-white">Camera {cameraId}</span>
        <div className={`w-2 h-2 rounded-full ${status === 'active' && !streamError ? 'bg-green-400' : 'bg-red-400'}`}></div>
      </div>
      <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
        {status === 'active' && !streamError ? (
          <>
            {isLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                <div className="text-center text-gray-500">
                  <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <div className="text-xs">Loading Stream...</div>
                </div>
              </div>
            )}
            <img
              src={`http://localhost:8000/api/cameras/${cameraId}/stream`}
              alt={`Camera ${cameraId} Stream`}
              className="w-full h-full object-cover"
              onLoad={handleImageLoad}
              onError={handleImageError}
              style={{ display: isLoading ? 'none' : 'block' }}
            />
          </>
        ) : (
          <div className="text-center text-gray-500">
            <Camera />
            <div className="text-xs mt-1">
              {streamError ? 'Stream Error' : status === 'active' ? 'Connecting...' : 'Offline'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Main App Component
const LiveWarehouse = () => {
  const [objects, setObjects] = useState([]);
  const [stats, setStats] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [warehouseConfig, setWarehouseConfig] = useState({ 
    width_feet: 180.0, 
    length_feet: 90.0, 
    width_meters: 54.864, 
    length_meters: 27.432 
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchObjects = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/tracking/objects');
      if (response.ok) {
        const data = await response.json();
        setObjects(data.objects || []);
        setError(null);
      } else {
        throw new Error('Failed to fetch');
      }
    } catch (err) {
      setError('Connection failed');
      setObjects([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/tracking/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Stats fetch failed:', err);
    }
  };

  const fetchCameras = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/cameras/status');
      if (response.ok) {
        const data = await response.json();
        setCameras(data.cameras || []);
      }
    } catch (err) {
      console.error('Camera status fetch failed:', err);
    }
  };

  const fetchWarehouseConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/warehouse/config');
      if (response.ok) {
        const data = await response.json();
        setWarehouseConfig({
          width_feet: data.width_feet || 180.0,
          length_feet: data.length_feet || 90.0,
          width_meters: data.width_meters || 54.864,
          length_meters: data.length_meters || 27.432
        });
      }
    } catch (err) {
      console.error('Warehouse config fetch failed:', err);
    }
  };

  useEffect(() => {
    fetchObjects();
    fetchStats();
    fetchCameras();
    fetchWarehouseConfig();
    const interval = setInterval(() => {
      fetchObjects();
      fetchStats();
      fetchCameras();
      // Fetch warehouse config less frequently (every 30 seconds)
    }, 3000);

    const configInterval = setInterval(fetchWarehouseConfig, 30000);

    return () => {
      clearInterval(interval);
      clearInterval(configInterval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Digital Twin - Live Warehouse Tracking</h1>
            <div className="flex gap-4 text-sm text-gray-400 mt-1">
              <span className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                Live CV System Connected
              </span>
              <span>Objects: {objects.length}</span>
              <span className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full"></div>
                Warehouse: 180ft × 90ft (45ft × 30ft Camera 8 active)
              </span>
              {error && <span className="text-red-400">Error: {error}</span>}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button className="p-2 hover:bg-gray-700 rounded-lg">
              <Search />
            </button>
            <button className="p-2 hover:bg-gray-700 rounded-lg">
              <Bell />
            </button>
            <button className="p-2 hover:bg-gray-700 rounded-lg">
              <User />
            </button>
          </div>
        </div>
      </header>

      <div className="flex h-screen">
        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Live Warehouse Map */}
          <div className="flex-1 p-4">
            <div className="bg-gray-800 rounded-lg p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold">Live Object Tracking</h2>
                <div className="flex items-center space-x-3 text-xs">
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-green-400 rounded animate-pulse"></div>
                    <span className="text-gray-400">Real-time Detection</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-4 h-3 bg-blue-400 bg-opacity-20 border border-blue-400 rounded-sm"></div>
                    <span className="text-gray-400">Object Bounding Boxes</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                    <span className="text-gray-400">Object Centers</span>
                  </div>
                </div>
              </div>

              {/* Warehouse visualization - Using REAL dimensions */}
              <div className="flex-1 flex justify-center items-center">
                <div className="relative">
                  {/* Warehouse boundary - Properly oriented: 180ft width (horizontal) × 90ft length (vertical) */}
                  <div
                    className="relative bg-gray-600 border-2 border-gray-500 rounded-lg"
                    style={{
                      width: `800px`, // Fixed width for better visualization
                      height: `400px`, // Height maintains 180:90 = 2:1 aspect ratio
                    }}
                  >
                    {/* Grid lines every 30ft */}
                    <div className="absolute inset-0 opacity-20">
                      {/* Vertical lines every 30ft */}
                      {Array.from({ length: Math.floor(180 / 30) + 1 }, (_, i) => (
                        <div key={`v-${i}`}>
                          <div
                            className="absolute h-full border-l border-gray-400"
                            style={{ left: `${(i * 30 / 180) * 100}%` }}
                          />
                          {i > 0 && (
                            <div
                              className="absolute text-xs text-gray-400 font-medium"
                              style={{ 
                                left: `${(i * 30 / 180) * 100}%`, 
                                top: '-20px',
                                transform: 'translateX(-50%)'
                              }}
                            >
                              {i * 30}ft
                            </div>
                          )}
                        </div>
                      ))}
                      {/* Horizontal lines every 30ft */}
                      {Array.from({ length: Math.floor(90 / 30) + 1 }, (_, i) => (
                        <div key={`h-${i}`}>
                          <div
                            className="absolute w-full border-t border-gray-400"
                            style={{ top: `${(i * 30 / 90) * 100}%` }}
                          />
                          {i > 0 && (
                            <div
                              className="absolute text-xs text-gray-400 font-medium"
                              style={{ 
                                left: '-30px', 
                                top: `${(i * 30 / 90) * 100}%`,
                                transform: 'translateY(-50%)'
                              }}
                            >
                              {i * 30}ft
                            </div>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Camera 8 coverage zone overlay - AT FAR END OF WAREHOUSE */}
                    <div
                      className="absolute border-2 border-green-400 bg-green-400 bg-opacity-10 rounded"
                      style={{
                        left: `${(0 / 180) * 100}%`, // Camera 8: x_start = 0ft
                        top: `${(60 / 90) * 100}%`,  // Camera 8: y_start = 60ft (far end)
                        width: `${(45 / 180) * 100}%`, // Camera 8: width = 45ft
                        height: `${(30 / 90) * 100}%`, // Camera 8: height = 30ft (90-60)
                      }}
                    >
                      <div className="absolute top-1 left-1 text-xs text-green-300 font-semibold">
                        Camera 8 Zone (Far End)
                      </div>
                    </div>

                    {/* Objects with Real Bounding Boxes */}
                    {objects.map((obj) => {
                      if (!obj.real_center || !obj.bbox) return null;

                      // CV system already returns global warehouse coordinates for Camera 8
                      // Camera 8 local coordinates (0-45ft, 0-30ft) are transformed to global (0-45ft, 50-80ft)
                      const globalX = obj.real_center[0]; // Global X coordinate (0-45ft range)
                      const globalY = obj.real_center[1]; // Global Y coordinate (50-80ft range) - NO OFFSET NEEDED

                      // Convert to percentage of full warehouse dimensions
                      const centerX = (globalX / 180) * 100; // Position within 180ft warehouse width
                      const centerY = (globalY / 90) * 100;  // Position within 90ft warehouse length

                      // Calculate bounding box dimensions based on Camera 8's coverage
                      const bboxWidth = obj.bbox[2] - obj.bbox[0]; // pixel width
                      const bboxHeight = obj.bbox[3] - obj.bbox[1]; // pixel height

                      // Scale bbox relative to Camera 8's coverage area (45ft × 30ft), not full warehouse
                      const realWidth = (bboxWidth / 640) * 45; // Scale to Camera 8's 45ft width
                      const realHeight = (bboxHeight / 480) * 30; // Scale to Camera 8's 30ft height

                      // Convert real dimensions to percentage of full warehouse display
                      const displayWidth = (realWidth / 180) * 100; // Relative to full 180ft width
                      const displayHeight = (realHeight / 90) * 100; // Relative to full 90ft height

                      return (
                        <div
                          key={obj.persistent_id}
                          className="absolute"
                          style={{
                            left: `${centerX - displayWidth/2}%`,
                            top: `${centerY - displayHeight/2}%`,
                            width: `${displayWidth}%`,
                            height: `${displayHeight}%`
                          }}
                        >
                          {/* Bounding Box */}
                          <div className="w-full h-full border-2 border-blue-400 bg-blue-400 bg-opacity-20 rounded-sm shadow-lg animate-pulse">
                            {/* Center dot */}
                            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-blue-600 rounded-full"></div>
                          </div>

                          {/* Object Label */}
                          <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-800 px-2 py-1 rounded text-white font-medium text-xs whitespace-nowrap">
                            ID: {obj.persistent_id}
                          </div>

                          {/* Coordinate and Size Info - Show actual global coordinates */}
                          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-80 whitespace-nowrap">
                            ({globalX.toFixed(1)}ft, {globalY.toFixed(1)}ft)
                            <br />
                            {realWidth.toFixed(1)}×{realHeight.toFixed(1)}ft
                          </div>

                          {/* Confidence indicator */}
                          <div className="absolute top-0 right-0 bg-green-600 text-white text-xs px-1 py-0.5 rounded-bl">
                            {(obj.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      );
                    })}

                    {/* Empty state */}
                    {objects.length === 0 && !loading && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center text-gray-400">
                          <div className="text-lg mb-2">
                            {error ? 'Connection Error' : 'Live CV System Active'}
                          </div>
                          <div className="text-sm">
                            {error ? 'Check MongoDB connection' : 'Monitoring warehouse - objects will appear when detected'}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Loading state */}
                    {loading && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center text-gray-400">
                          <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                          <div>Connecting to CV System...</div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Labels - Real warehouse dimensions */}
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-gray-300 text-base font-medium">
                    180ft width
                  </div>
                  <div className="absolute -left-24 top-1/2 transform -translate-y-1/2 -rotate-90 text-gray-300 text-base font-medium">
                    90ft length
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Camera Feeds */}
          <div className="p-4 pt-0">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold">Live Camera Feeds</h3>
                <div className="text-xs text-gray-400">
                  {cameras.filter(c => c.status === 'active').length} of {cameras.length} cameras active
                </div>
              </div>
              <div className="grid grid-cols-4 gap-4">
                {cameras.length > 0 ? (
                  cameras.map((camera) => (
                    <CameraFeed
                      key={camera.camera_id}
                      cameraId={camera.camera_id}
                      status={camera.status}
                    />
                  ))
                ) : (
                  // Fallback while loading camera status
                  [1, 2, 3, 4].map((id) => (
                    <CameraFeed key={id} cameraId={id} status="offline" />
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 p-4 space-y-6">
          {/* Stats */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Live Analytics</h3>
            <div className="space-y-3">
              <StatCard
                icon={<Package />}
                label="Tracked Objects"
                value={stats?.unique_objects || objects.length}
                color="blue"
              />
              <StatCard
                icon={<Activity />}
                label="Total Detections"
                value={stats?.total_detections || 0}
                color="green"
              />
              <StatCard
                icon={<TrendingUp />}
                label="Recent Activity"
                value={stats?.recent_objects || 0}
                color="amber"
              />
              <StatCard
                icon={<Clock />}
                label="System Uptime"
                value="24h"
                color="green"
              />
            </div>
          </div>

          {/* System Status */}
          <div>
            <h3 className="text-lg font-semibold mb-4">System Status</h3>
            <div className="space-y-3">
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">CV Detection</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-xs text-green-400">Active</span>
                  </div>
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">MongoDB</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-xs text-green-400">Connected</span>
                  </div>
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Real-time Updates</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-xs text-green-400">Live</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Objects */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Recent Objects</h3>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {objects.slice(0, 5).map((obj) => (
                <div key={obj.persistent_id} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">ID: {obj.persistent_id}</span>
                    <span className="text-xs text-gray-400">
                      {obj.confidence ? `${(obj.confidence * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                  {obj.real_center && (
                    <div className="text-xs text-gray-400 mt-1">
                      Position: ({obj.real_center[0]?.toFixed(1)}ft, {obj.real_center[1]?.toFixed(1)}ft)
                    </div>
                  )}
                </div>
              ))}
              {objects.length === 0 && (
                <div className="text-center text-gray-500 py-4">
                  <div className="text-sm">No objects detected</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <LiveWarehouse />
  </React.StrictMode>,
)