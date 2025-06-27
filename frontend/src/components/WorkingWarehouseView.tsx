import React, { useState, useEffect } from 'react';

interface TrackedObject {
  persistent_id: number;
  center: { x: number; y: number };
  real_center?: [number, number] | null;
  confidence: number;
  age_seconds: number;
  times_seen: number;
  status?: string;
}

interface WarehouseConfig {
  width_feet?: number;
  length_feet?: number;
  width_meters: number;
  length_meters: number;
  calibrated: boolean;
}

const WorkingWarehouseView: React.FC = () => {
  const [objects, setObjects] = useState<TrackedObject[]>([]);
  const [config] = useState<WarehouseConfig>({
    width_meters: 10.0,
    length_meters: 8.0,
    calibrated: true
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCameraZones, setShowCameraZones] = useState(true);

  // Camera zones for Phase 1 (Column 3 active)
  const cameraZones = [
    { id: 8, name: "Camera 8", x_start: 120, x_end: 180, y_start: 0, y_end: 25, active: true },
    { id: 9, name: "Camera 9", x_start: 120, x_end: 180, y_start: 25, y_end: 50, active: true },
    { id: 10, name: "Camera 10", x_start: 120, x_end: 180, y_start: 50, y_end: 75, active: true },
    { id: 11, name: "Camera 11", x_start: 120, x_end: 180, y_start: 75, y_end: 90, active: true },
    // Standby cameras
    { id: 5, name: "Camera 5", x_start: 60, x_end: 120, y_start: 0, y_end: 22.5, active: false },
    { id: 6, name: "Camera 6", x_start: 60, x_end: 120, y_start: 22.5, y_end: 45, active: false },
    { id: 7, name: "Camera 7", x_start: 60, x_end: 120, y_start: 45, y_end: 67.5, active: false },
    { id: 1, name: "Camera 1", x_start: 0, x_end: 60, y_start: 0, y_end: 22.5, active: false },
    { id: 2, name: "Camera 2", x_start: 0, x_end: 60, y_start: 22.5, y_end: 45, active: false },
    { id: 3, name: "Camera 3", x_start: 0, x_end: 60, y_start: 45, y_end: 67.5, active: false },
    { id: 4, name: "Camera 4", x_start: 0, x_end: 60, y_start: 67.5, y_end: 90, active: false },
  ];

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
      setError('Unable to connect to tracking system');
      setObjects([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchObjects();
    const interval = setInterval(fetchObjects, 3000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status?: string, age_seconds?: number) => {
    if (status) {
      switch (status) {
        case 'new': return '#ffd93d';
        case 'tracking': return '#6bcf7f';
        case 'established': return '#4ecdc4';
        default: return '#ff6b6b';
      }
    }
    // Fallback based on age
    if (age_seconds && age_seconds < 5) return '#ffd93d';
    if (age_seconds && age_seconds < 60) return '#6bcf7f';
    return '#4ecdc4';
  };

  const formatAge = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  };

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-xl p-8 h-full flex items-center justify-center border border-gray-700 shadow-2xl">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-800 rounded-full flex items-center justify-center border-2 border-gray-600">
            <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
          </div>
          <div className="text-xl font-semibold text-white mb-2">Loading Warehouse Data</div>
          <div className="text-gray-400">Connecting to tracking system...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl p-8 h-full border border-gray-700 shadow-2xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-3xl font-bold text-white tracking-tight">Live Object Tracking</h2>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-gray-800 px-3 py-1.5 rounded-lg border border-gray-600">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-300 font-medium">Live CV System</span>
            </div>
            <div className="bg-gray-800 px-3 py-1.5 rounded-lg border border-gray-600">
              <span className="text-sm text-blue-400 font-semibold">{objects.length} Objects</span>
            </div>
            <div className="bg-gray-800 px-3 py-1.5 rounded-lg border border-gray-600">
              <span className="text-xs text-green-400 font-medium">🔗 MongoDB</span>
            </div>
            <button
              onClick={() => setShowCameraZones(!showCameraZones)}
              className={`px-3 py-1.5 rounded-lg border text-sm font-medium transition-colors ${
                showCameraZones
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-gray-800 border-gray-600 text-gray-300 hover:bg-gray-700'
              }`}
            >
              📹 {showCameraZones ? 'Hide' : 'Show'} Zones
            </button>
          </div>
        </div>
        <div className="flex gap-6 text-sm text-gray-400">
          <span className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-gray-500 rounded-full"></div>
            Warehouse: {config.width_meters}m × {config.length_meters}m
          </span>
          <span className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
            Live CV Detection Active
          </span>
          <span className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-blue-400 rounded-full"></div>
            Real-time Coordinates
          </span>
          {error && (
            <span className="flex items-center gap-2 text-red-400">
              <div className="w-1.5 h-1.5 bg-red-400 rounded-full"></div>
              {error}
            </span>
          )}
        </div>
      </div>

      {/* Warehouse visualization */}
      <div className="flex justify-center items-center" style={{ minHeight: '500px' }}>
        <div className="relative">
          {/* Dimension labels */}
          <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-gray-400 text-sm font-medium bg-gray-800 px-3 py-1 rounded-full border border-gray-600">
            {config.width_feet || (config.width_meters * 3.28084).toFixed(0)}ft
          </div>
          <div className="absolute -left-20 top-1/2 transform -translate-y-1/2 -rotate-90 text-gray-400 text-sm font-medium bg-gray-800 px-3 py-1 rounded-full border border-gray-600">
            {config.length_feet || (config.length_meters * 3.28084).toFixed(0)}ft
          </div>

          {/* Warehouse boundary - Modern gray background */}
          <div
            className="relative bg-gray-600 border-2 border-gray-500 rounded-lg shadow-inner"
            style={{
              width: '700px',
              height: `${((config.length_feet || config.length_meters * 3.28084) / (config.width_feet || config.width_meters * 3.28084)) * 700}px`,
              maxHeight: '560px',
              minHeight: '350px'
            }}
          >
            {/* Subtle grid lines for reference */}
            <div className="absolute inset-0 opacity-15">
              {Array.from({ length: Math.floor((config.width_feet || config.width_meters * 3.28084) / 30) + 1 }, (_, i) => (
                <div
                  key={`v-${i}`}
                  className="absolute h-full border-l border-gray-400"
                  style={{ left: `${(i * 30 / (config.width_feet || config.width_meters * 3.28084)) * 100}%` }}
                />
              ))}
              {Array.from({ length: Math.floor((config.length_feet || config.length_meters * 3.28084) / 30) + 1 }, (_, i) => (
                <div
                  key={`h-${i}`}
                  className="absolute w-full border-t border-gray-400"
                  style={{ top: `${(i * 30 / (config.length_feet || config.length_meters * 3.28084)) * 100}%` }}
                />
              ))}
            </div>

            {/* Corner markers for warehouse boundaries */}
            <div className="absolute top-2 left-2 w-3 h-3 border-l-2 border-t-2 border-gray-400 opacity-60"></div>
            <div className="absolute top-2 right-2 w-3 h-3 border-r-2 border-t-2 border-gray-400 opacity-60"></div>
            <div className="absolute bottom-2 left-2 w-3 h-3 border-l-2 border-b-2 border-gray-400 opacity-60"></div>
            <div className="absolute bottom-2 right-2 w-3 h-3 border-r-2 border-b-2 border-gray-400 opacity-60"></div>

            {/* Camera zones */}
            {showCameraZones && cameraZones.map((zone) => {
              const warehouseWidthFt = config.width_feet || config.width_meters * 3.28084;
              const warehouseLengthFt = config.length_feet || config.length_meters * 3.28084;

              // Calculate zone position and size - FIXED: Flipped mapping
              const x = ((warehouseWidthFt - zone.x_end) / warehouseWidthFt) * 100; // Flipped mapping
              const y = (zone.y_start / warehouseLengthFt) * 100;
              const width = ((zone.x_end - zone.x_start) / warehouseWidthFt) * 100;
              const height = ((zone.y_end - zone.y_start) / warehouseLengthFt) * 100;

              return (
                <div
                  key={zone.id}
                  className={`absolute border-2 transition-all duration-200 ${
                    zone.active
                      ? 'border-green-400 bg-green-500 bg-opacity-20 hover:bg-opacity-30'
                      : 'border-gray-500 bg-gray-600 bg-opacity-15 hover:bg-opacity-25'
                  }`}
                  style={{
                    left: `${x}%`,
                    top: `${y}%`,
                    width: `${width}%`,
                    height: `${height}%`
                  }}
                  title={`${zone.name} - ${zone.x_start}-${zone.x_end}ft × ${zone.y_start}-${zone.y_end}ft`}
                >
                  {/* Camera ID badge */}
                  <div className={`absolute top-1 left-1 text-xs font-bold px-1.5 py-0.5 rounded ${
                    zone.active
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-600 text-white'
                  }`}>
                    {zone.id}
                  </div>

                  {/* Active indicator */}
                  {zone.active && (
                    <div className="absolute top-1 right-1 w-2 h-2 bg-green-400 rounded-full animate-pulse">
                      <div className="absolute inset-0 bg-green-400 rounded-full animate-ping"></div>
                    </div>
                  )}

                  {/* Zone coordinates */}
                  <div className="absolute bottom-1 right-1 text-xs bg-black bg-opacity-75 text-white px-1 py-0.5 rounded">
                    {zone.x_start}-{zone.x_end}×{zone.y_start}-{zone.y_end}ft
                  </div>
                </div>
              );
            })}

            {/* Modern object markers */}
            {objects.map((object) => {
              if (!object.real_center || !Array.isArray(object.real_center) || object.real_center.length < 2) return null;

              const warehouseWidthFt = config.width_feet || config.width_meters * 3.28084;
              const warehouseLengthFt = config.length_feet || config.length_meters * 3.28084;

              // FIXED: Flipped mapping so Camera 8 (120-180ft) appears on YOUR LEFT side
              const x = ((warehouseWidthFt - object.real_center[0]) / warehouseWidthFt) * 100; // Flipped mapping
              const y = (object.real_center[1] / warehouseLengthFt) * 100; // Y-axis correct
              const color = getStatusColor(object.status, object.age_seconds);

              return (
                <div
                  key={object.persistent_id}
                  className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
                  style={{ left: `${x}%`, top: `${y}%` }}
                >
                  {/* Outer glow ring */}
                  <div
                    className="absolute w-6 h-6 rounded-full opacity-30 animate-pulse"
                    style={{
                      backgroundColor: color,
                      transform: 'translate(-50%, -50%)',
                      left: '50%',
                      top: '50%'
                    }}
                  />

                  {/* Main marker dot */}
                  <div
                    className="relative w-4 h-4 rounded-full border-2 border-white shadow-lg transition-all duration-200 group-hover:scale-125"
                    style={{ backgroundColor: color }}
                  />

                  {/* ID label with modern styling */}
                  <div className="absolute top-5 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-0.5 rounded-full font-semibold border border-gray-600 shadow-lg">
                    {object.persistent_id}
                  </div>

                  {/* Modern tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-3 opacity-0 group-hover:opacity-100 transition-all duration-200 bg-gray-900 text-white text-xs rounded-lg px-3 py-2 whitespace-nowrap z-20 border border-gray-600 shadow-xl">
                    <div className="space-y-1">
                      <div className="flex justify-between gap-3">
                        <span className="text-gray-400">ID:</span>
                        <span className="font-semibold">{object.persistent_id}</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-gray-400">Age:</span>
                        <span>{formatAge(object.age_seconds)}</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="text-green-400">{(object.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-gray-400">Position:</span>
                        <span className="text-blue-400">({object.real_center[0].toFixed(1)}ft, {object.real_center[1].toFixed(1)}ft)</span>
                      </div>
                    </div>
                    {/* Tooltip arrow */}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                  </div>
                </div>
              );
            })}

            {/* Modern empty state */}
            {objects.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gray-700 rounded-full flex items-center justify-center border-2 border-gray-500">
                    <div className="w-8 h-8 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                  <div className="text-lg font-semibold text-gray-300 mb-2">
                    {error ? 'Connection Error' : 'Live CV System Active'}
                  </div>
                  <div className="text-sm text-gray-500">
                    {error ? 'Check MongoDB connection and try again' : 'Monitoring warehouse - objects will appear when detected'}
                  </div>
                  {!error && (
                    <div className="text-xs text-green-400 mt-2">
                      🔗 Connected to MongoDB • Real-time tracking enabled
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Coordinate indicators - FIXED for flipped mapping */}
          <div className="absolute -bottom-8 left-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            ({config.width_feet || (config.width_meters * 3.28084).toFixed(0)}ft, {config.length_feet || (config.length_meters * 3.28084).toFixed(0)}ft)
          </div>
          <div className="absolute -bottom-8 right-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            (0, {config.length_feet || (config.length_meters * 3.28084).toFixed(0)}ft)
          </div>
          <div className="absolute -top-8 left-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            ({config.width_feet || (config.width_meters * 3.28084).toFixed(0)}ft, 0)
          </div>
          <div className="absolute -top-8 right-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            (0, 0) - Origin
          </div>
        </div>
      </div>

      {/* Modern legend */}
      <div className="mt-8 flex justify-center">
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-600 shadow-lg">
          <h3 className="text-sm font-semibold text-white mb-4 text-center">Object Status Legend</h3>
          <div className="flex gap-6 text-sm">
            <div className="flex items-center gap-3 bg-gray-700 px-4 py-2 rounded-lg border border-gray-600">
              <div className="relative">
                <div className="w-4 h-4 rounded-full bg-yellow-400 border-2 border-white shadow-lg"></div>
                <div className="absolute w-6 h-6 rounded-full bg-yellow-400 opacity-20 animate-pulse" style={{ transform: 'translate(-25%, -25%)' }}></div>
              </div>
              <span className="text-gray-300 font-medium">New (&lt;5s)</span>
            </div>
            <div className="flex items-center gap-3 bg-gray-700 px-4 py-2 rounded-lg border border-gray-600">
              <div className="relative">
                <div className="w-4 h-4 rounded-full bg-green-400 border-2 border-white shadow-lg"></div>
                <div className="absolute w-6 h-6 rounded-full bg-green-400 opacity-20 animate-pulse" style={{ transform: 'translate(-25%, -25%)' }}></div>
              </div>
              <span className="text-gray-300 font-medium">Tracking (5s-1m)</span>
            </div>
            <div className="flex items-center gap-3 bg-gray-700 px-4 py-2 rounded-lg border border-gray-600">
              <div className="relative">
                <div className="w-4 h-4 rounded-full bg-teal-400 border-2 border-white shadow-lg"></div>
                <div className="absolute w-6 h-6 rounded-full bg-teal-400 opacity-20 animate-pulse" style={{ transform: 'translate(-25%, -25%)' }}></div>
              </div>
              <span className="text-gray-300 font-medium">Established (&gt;1m)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkingWarehouseView;
