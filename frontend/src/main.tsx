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
              src={`${import.meta.env.VITE_API_BASE_URL}/api/cameras/${cameraId}/stream`}
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

// Object Details Sidebar Component
const ObjectDetailsSidebar = ({ object, isOpen, onClose }) => {
  if (!isOpen || !object) return null;

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      // Return exact timestamp in readable format
      return date.toLocaleString();
    } catch {
      return 'Invalid date';
    }
  };

  return (
    <div className={`fixed right-0 top-0 h-full w-80 bg-gray-800 border-l border-gray-700 transform transition-transform duration-300 ease-in-out z-50 ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Pallet Details</h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div className="p-4 overflow-y-auto h-full pb-20">
        <div className="space-y-4">
          {/* Object ID */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Object ID</div>
            <div className="text-xl font-bold text-blue-400">
              {object.persistent_id || object.global_id || 'Unknown'}
            </div>
          </div>

          {/* Warp ID */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">🏷️ Warp ID</div>
            <div className="flex items-center gap-2">
              {object.warp_id ? (
                <>
                  <span className="text-green-400 font-mono font-semibold bg-green-900/20 px-2 py-1 rounded text-sm">
                    {object.warp_id}
                  </span>
                  <div className="w-2 h-2 bg-green-400 rounded-full" title="QR Code Linked"></div>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" title="Awaiting Robot Assignment"></div>
                  <span className="text-orange-400 font-medium">Not assigned yet</span>
                </>
              )}
            </div>
          </div>

          {/* Camera Info */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Camera</div>
            <div className="text-white font-medium">Zone {object.camera_id || 'Unknown'}</div>
          </div>

          {/* Timing Information */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="space-y-3">
              <div>
                <div className="text-sm text-gray-400 mb-1">Inbound Time</div>
                <div className="text-white text-sm">
                  {formatTimestamp(object.first_seen || object.timestamp)}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Last Seen</div>
                <div className="text-white text-sm">
                  {formatTimestamp(object.last_seen || object.timestamp)}
                </div>
              </div>
              {object.warp_id_linked_at && (
                <div>
                  <div className="text-sm text-gray-400 mb-1">Warp Linked</div>
                  <div className="text-purple-400 text-sm">
                    {formatTimestamp(object.warp_id_linked_at)}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Detection Quality */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">🎯 Detection Quality</div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-300">Confidence:</span>
                <span className="text-green-400 font-medium">
                  {object.confidence ? `${(object.confidence * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Area:</span>
                <span className="text-white">{object.area?.toLocaleString() || 'N/A'} px</span>
              </div>
            </div>
          </div>



          {/* Coordinates */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">📍 Coordinates</div>
            <div className="space-y-2">

              <div className="flex justify-between">
                <span className="text-gray-300">Physical:</span>
                <span className="text-cyan-400 font-medium">
                  ({object.physical_x_ft?.toFixed(1) || 'N/A'}, {object.physical_y_ft?.toFixed(1) || 'N/A'}) ft
                </span>
              </div>
            </div>
          </div>

          {/* Tracking Info */}
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">🔄 Tracking</div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-300">Times Seen:</span>
                <span className="text-white">{object.times_seen || 1}</span>
              </div>
              {object.similarity_score && (
                <div className="flex justify-between">
                  <span className="text-gray-300">Similarity:</span>
                  <span className="text-blue-400 font-medium">
                    {(object.similarity_score * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Raw Data (Collapsible) */}
          <details className="bg-gray-700 rounded-lg">
            <summary className="p-3 cursor-pointer text-sm text-gray-400 hover:text-white">
              🔧 Raw Data
            </summary>
            <div className="px-3 pb-3">
              <pre className="text-xs text-gray-300 bg-gray-800 p-2 rounded overflow-x-auto">
                {JSON.stringify(object, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
};

// Helper function to get object color from database
const getObjectColor = (obj) => {
  // Priority 1: Use RGB values if available
  if (obj.color_rgb && Array.isArray(obj.color_rgb) && obj.color_rgb.length >= 3) {
    return `rgb(${obj.color_rgb[0]}, ${obj.color_rgb[1]}, ${obj.color_rgb[2]})`;
  }

  // Priority 2: Use hex color if available
  if (obj.color_hex) {
    return obj.color_hex;
  }

  // Priority 3: Map color names to hex values
  if (obj.color_name && obj.color_confidence && obj.color_confidence > 0.3) {
    const colorMap = {
      'red': '#ff4444',
      'orange': '#ff8800',
      'yellow': '#ffdd00',
      'green': '#44ff44',
      'blue': '#4444ff',
      'purple': '#8844ff',
      'pink': '#ff44aa',
      'brown': '#8b4513',
      'black': '#333333',
      'white': '#f0f0f0',
      'gray': '#888888',
      'grey': '#888888',
      'dark': '#444444'
    };

    const detectedColor = colorMap[obj.color_name.toLowerCase()];
    if (detectedColor) {
      return detectedColor;
    }
  }

  // Fallback: amber color
  return '#d97706';
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
  const [selectedObject, setSelectedObject] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isPolling, setIsPolling] = useState(false); // Prevent overlapping requests
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightedObject, setHighlightedObject] = useState(null);

  // Handle object selection
  const handleObjectClick = (obj) => {
    setSelectedObject(obj);
    setSidebarOpen(true);
  };

  // Handle search functionality
  const handleSearch = (query) => {
    setSearchQuery(query);

    if (!query.trim()) {
      setHighlightedObject(null);
      return;
    }

    // Search for objects by persistent_id or global_id
    const foundObject = objects.find(obj =>
      obj.persistent_id?.toString().includes(query.trim()) ||
      obj.global_id?.toString().includes(query.trim())
    );

    if (foundObject) {
      setHighlightedObject(foundObject);
      // Auto-select the found object to show details
      setSelectedObject(foundObject);
      setSidebarOpen(true);
    } else {
      setHighlightedObject(null);
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setHighlightedObject(null);
  };

  // Handle clicking outside to deselect
  const handleBackgroundClick = () => {
    setSelectedObject(null);
    setSidebarOpen(false);
  };

  // Handle ESC key to close sidebar
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && sidebarOpen) {
        setSelectedObject(null);
        setSidebarOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [sidebarOpen]);

  const fetchObjects = async () => {
    // Prevent overlapping requests
    if (isPolling) {
      console.log('⏭️ Skipping fetch - previous request still in progress');
      return;
    }

    setIsPolling(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/tracking/objects`);
      if (response.ok) {
        const data = await response.json();
        setObjects(data.objects || []);
        setError(null);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (err) {
      console.error('❌ Fetch error:', err);
      setError('Connection failed');
      setObjects([]);
    } finally {
      setLoading(false);
      setIsPolling(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/tracking/stats`);
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
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/cameras/status`);
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
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/warehouse/config`);
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

    // Reduced polling frequency to prevent overlapping requests
    const interval = setInterval(() => {
      fetchObjects();
      fetchStats();
    }, 2000); // Changed from 500ms to 2000ms (2 seconds)

    // Fetch cameras less frequently since they don't change often
    const cameraInterval = setInterval(fetchCameras, 10000); // Every 10 seconds

    // Fetch warehouse config even less frequently
    const configInterval = setInterval(fetchWarehouseConfig, 30000); // Every 30 seconds

    return () => {
      clearInterval(interval);
      clearInterval(cameraInterval);
      clearInterval(configInterval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          {/* Logo Section */}
          <div className="flex items-center">
            <img
              src="/logo3.png"
              alt="Logo"
              className="h-12 w-auto mr-4"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>

          {/* Centered Title Section */}
          <div className="flex-1 flex flex-col items-center">
            <h1 className="text-2xl font-bold text-center">Digital Twin - Live Warehouse Tracking</h1>
            <div className="flex gap-4 text-sm text-gray-400 mt-1">
              <span className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                Live CV System Connected
              </span>
              <span>Objects: {objects.length}{selectedObject ? ` | Selected: ${selectedObject.persistent_id || selectedObject.global_id}` : ''}</span>
              <span className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full"></div>
                Warehouse: 180ft × 90ft
              </span>
              {error && <span className="text-red-400">Error: {error}</span>}
            </div>
          </div>

          {/* Right Side Buttons */}
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
              <div className="flex items-center justify-end mb-3">
                <div className="flex items-center space-x-3 text-xs">
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-3 bg-green-400 rounded animate-pulse"></div>
                    <span className="text-gray-400">Active Camera Zone</span>
                  </div>

                </div>
              </div>

              {/* Warehouse visualization - Using REAL dimensions */}
              <div className="flex-1 flex justify-center items-center">
                <div className="relative">
                  {/* Warehouse boundary - Properly oriented: 180ft width (horizontal) × 90ft length (vertical) */}
                  <div
                    className="relative bg-gray-600 border-2 border-gray-500 rounded-lg cursor-pointer"
                    style={{
                      width: `800px`, // Fixed width for better visualization
                      height: `400px`, // Height maintains 180:90 = 2:1 aspect ratio
                    }}
                    onClick={handleBackgroundClick}
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

                    {/* Camera zones - Phase 1: Column 3 active */}
                    {[
                      { id: 8, x_start: 120, x_end: 180, y_start: 0, y_end: 22.5, active: true },
                      { id: 9, x_start: 120, x_end: 180, y_start: 22.5, y_end: 45, active: true },
                      { id: 10, x_start: 120, x_end: 180, y_start: 45, y_end: 67.5, active: true },
                      { id: 11, x_start: 120, x_end: 180, y_start: 67.5, y_end: 90, active: true },
                      // Standby cameras
                      { id: 5, x_start: 60, x_end: 120, y_start: 0, y_end: 22.5, active: true },
                      { id: 6, x_start: 60, x_end: 120, y_start: 22.5, y_end: 45, active: true },
                      { id: 7, x_start: 60, x_end: 120, y_start: 45, y_end: 67.5, active: true },
                      { id: 1, x_start: 0, x_end: 60, y_start: 0, y_end: 22.5, active: true },
                      { id: 2, x_start: 0, x_end: 60, y_start: 22.5, y_end: 45, active: true },
                      { id: 3, x_start: 0, x_end: 60, y_start: 45, y_end: 67.5, active: true },
                      { id: 4, x_start: 0, x_end: 60, y_start: 67.5, y_end: 90, active: true },
                    ].map((zone) => (
                      <div
                        key={zone.id}
                        className={`absolute border-2 ${
                          zone.active
                            ? 'border-green-400 bg-green-400 bg-opacity-15'
                            : 'border-gray-400 bg-gray-400 bg-opacity-10'
                        }`}
                        style={{
                          left: `${((180 - zone.x_end) / 180) * 100}%`, // Flipped mapping
                          top: `${(zone.y_start / 90) * 100}%`,
                          width: `${((zone.x_end - zone.x_start) / 180) * 100}%`,
                          height: `${((zone.y_end - zone.y_start) / 90) * 100}%`,
                        }}
                        title={`Camera ${zone.id} - ${zone.x_start}-${zone.x_end}ft × ${zone.y_start}-${zone.y_end}ft`}
                      >
                        <div className={`absolute top-1 left-1 text-xs px-1.5 py-0.5 rounded font-bold ${
                          zone.active
                            ? 'bg-green-600 text-white'
                            : 'bg-gray-600 text-white'
                        }`}>
                          {zone.id}
                        </div>
                        {zone.active && (
                          <div className="absolute top-1 right-1 w-2 h-2 bg-green-400 rounded-full animate-pulse">
                            <div className="absolute inset-0 bg-green-400 rounded-full animate-ping"></div>
                          </div>
                        )}
                      </div>
                    ))}

                    {/* SVG Layer for Quadrangle Shapes */}
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ zIndex: 5 }}
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                    >
                      {/* Define glow filter for search highlighting */}
                      <defs>
                        <filter id="searchGlow" x="-50%" y="-50%" width="200%" height="200%">
                          <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#3b82f6" floodOpacity="0.8"/>
                          <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#3b82f6" floodOpacity="0.4"/>
                        </filter>
                      </defs>
                      {objects.map((obj) => {
                        // Check if object has 4-corner data for quadrangle rendering
                        if (obj.physical_corners && obj.physical_corners.length === 4) {
                          // Convert all 4 physical corners to screen coordinates
                          const screenCorners = obj.physical_corners.map(corner => {
                            const x = ((180 - corner[0]) / 180) * 100; // Flipped mapping
                            const y = (corner[1] / 90) * 100;
                            return [x, y];
                          });

                          // Create SVG polygon points string
                          const points = screenCorners.map(corner => `${corner[0]},${corner[1]}`).join(' ');

                          // Get object color from database
                          const objectColor = getObjectColor(obj);

                          const isSelected = selectedObject?.persistent_id === obj.persistent_id;
                          const isSearchHighlighted = highlightedObject?.persistent_id === obj.persistent_id;

                          return (
                            <React.Fragment key={`quad-${obj.persistent_id}`}>
                              <polygon
                                points={points}
                                fill={objectColor}
                                stroke={isSelected ? "#3b82f6" : "rgba(255,255,255,0.3)"}
                                strokeWidth={isSelected ? "0.3" : "0.2"}
                                opacity={isSelected || isSearchHighlighted ? 1 : 0.85}
                                filter={isSearchHighlighted ? "url(#searchGlow)" : "none"}
                                className={isSearchHighlighted ? "animate-pulse" : ""}
                                style={{ cursor: 'pointer', pointerEvents: 'auto' }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleObjectClick(obj);
                                }}
                              />
                              {/* Red center dot for search highlighting */}
                              {isSearchHighlighted && (
                                <circle
                                  cx={screenCorners.reduce((sum, corner) => sum + corner[0], 0) / screenCorners.length}
                                  cy={screenCorners.reduce((sum, corner) => sum + corner[1], 0) / screenCorners.length}
                                  r="1"
                                  fill="#ef4444"
                                  className="animate-pulse"
                                  style={{ pointerEvents: 'none' }}
                                />
                              )}
                            </React.Fragment>
                          );
                        }
                        return null;
                      })}
                    </svg>

                    {/* Objects as Simple Points (Fallback for objects without quadrangle data) */}
                    {objects.map((obj) => {
                      // Skip if object has quadrangle data (already rendered as SVG)
                      if (obj.physical_corners && obj.physical_corners.length === 4) {
                        return null;
                      }

                      // Use real_center_x and real_center_y if available, fallback to real_center array
                      const globalX = obj.real_center_x !== undefined ? obj.real_center_x : (obj.real_center ? obj.real_center[0] : 0);
                      const globalY = obj.real_center_y !== undefined ? obj.real_center_y : (obj.real_center ? obj.real_center[1] : 0);

                      if (globalX === 0 && globalY === 0) return null;

                      // Convert to percentage of warehouse dimensions for display
                      // FIXED: Flipped mapping so Camera 8 (120-180ft) appears on YOUR LEFT side
                      const centerX = ((180 - globalX) / 180) * 100; // Flipped: Camera 8 → low % → LEFT side
                      const centerY = (globalY / 90) * 100;  // Y-axis: top-to-bottom

                      const isSelected = selectedObject?.persistent_id === obj.persistent_id;
                      const isSearchHighlighted = highlightedObject?.persistent_id === obj.persistent_id;

                      return (
                        <div
                          key={obj.persistent_id}
                          className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer"
                          style={{
                            left: `${centerX}%`,
                            top: `${centerY}%`,
                            zIndex: isSelected ? 20 : 10
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleObjectClick(obj);
                          }}
                        >
                          {/* Clean Object Box with Real Color */}
                          <div
                            className={`w-8 h-8 border rounded-sm transition-all duration-200 hover:scale-110 ${
                              isSelected
                                ? 'border-blue-400 border-4 scale-110'
                                : 'border-black hover:border-gray-600'
                            } ${isSearchHighlighted ? 'animate-pulse' : ''}`}
                            style={{
                              backgroundColor: getObjectColor(obj),
                              opacity: isSelected || isSearchHighlighted ? 1 : 0.85,
                              boxShadow: isSearchHighlighted
                                ? '0 0 15px 3px rgba(59, 130, 246, 0.6), 0 0 25px 5px rgba(59, 130, 246, 0.3)'
                                : 'none'
                            }}
                          >
                            {/* Red center dot for search highlighting */}
                            {isSearchHighlighted && (
                              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                              </div>
                            )}
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

          {/* Camera Feeds - HIDDEN FOR NOW - Will be enabled later with RTSP integration */}
        </div>

        {/* Object Details Sidebar */}
        <ObjectDetailsSidebar
          object={selectedObject}
          isOpen={sidebarOpen}
          onClose={() => {
            setSelectedObject(null);
            setSidebarOpen(false);
          }}
        />

        {/* Right Sidebar */}
        <div className={`w-80 bg-gray-800 border-l border-gray-700 p-4 space-y-6 transition-all duration-300 ${sidebarOpen ? 'mr-80' : ''}`}>
          {/* Search */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Search Pallets</h3>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Enter Pallet ID (e.g., 8001)"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {searchQuery && (
                <button
                  onClick={clearSearch}
                  className="px-3 py-2 bg-gray-600 hover:bg-gray-500 rounded-md text-white transition-colors"
                  title="Clear search"
                >
                  ✕
                </button>
              )}
            </div>
            {searchQuery && !highlightedObject && (
              <div className="mt-2 text-sm text-red-400">
                No pallet found with ID "{searchQuery}"
              </div>
            )}
            {highlightedObject && (
              <div className="mt-2 text-sm text-green-400">
                Found: Pallet {highlightedObject.persistent_id || highlightedObject.global_id}
              </div>
            )}
          </div>

          {/* Stats */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Live Analytics</h3>
            <div className="space-y-3">
              <StatCard
                icon={<Package />}
                label="Tracked Pallets"
                value={stats?.unique_objects || objects.length}
                trend={null}
                trendLabel=""
                color="blue"
              />
              <StatCard
                icon={<Activity />}
                label="Space Utilization"
                value="60%"
                trend={null}
                trendLabel=""
                color="green"
              />
              <StatCard
                icon={<TrendingUp />}
                label="Recent Activity"
                value={stats?.recent_objects || 0}
                trend={null}
                trendLabel=""
                color="amber"
              />
              <StatCard
                icon={<Clock />}
                label="System Uptime"
                value="24h"
                trend={null}
                trendLabel=""
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