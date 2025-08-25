import React, { useState, useEffect } from 'react';
import { Camera } from 'lucide-react';
import ObjectCropsGallery from './ObjectCropsGallery';


interface TrackedObject {
  persistent_id: number;
  global_id?: number;
  warp_id?: string | null;  // NEW: Warp ID from QR code
  camera_id?: number;
  real_center: [number, number];  // Required - physical coordinates in feet
  bbox?: number[];  // [x1, y1, x2, y2] bounding box coordinates
  corners?: number[][];  // 4-point pixel coordinates [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
  physical_corners?: number[][];  // 4-point physical coordinates in feet
  shape_type?: string;  // 'quadrangle' or 'rectangle'
  confidence: number;
  area?: number;
  age_seconds: number;
  times_seen: number;
  status?: string;
  color_rgb?: number[];
  color_hsv?: number[];
  color_hex?: string;
  color_name?: string;
  color_confidence?: number;
  extraction_method?: string;
  warp_id_linked_at?: string;  // NEW: When Warp ID was linked
  first_seen?: string;  // NEW: First detection timestamp
  last_seen?: string;   // NEW: Last detection timestamp
}

interface WarehouseConfig {
  width_feet?: number;
  length_feet?: number;
  width_meters: number;
  length_meters: number;
  calibrated: boolean;
}

interface WWVProps {
  externalSearchQuery?: string;
  feedsEnabled?: boolean;
  onToggleFeeds?: (value: boolean) => void;
}
const WorkingWarehouseView: React.FC<WWVProps> = ({ externalSearchQuery, feedsEnabled, onToggleFeeds }) => {
  const [objects, setObjects] = useState<TrackedObject[]>([]);
  const [config] = useState<WarehouseConfig>({
    width_feet: 180.0,
    length_feet: 100.0,
    width_meters: 54.86,  // 180ft converted to meters
    length_meters: 30.48, // 100ft converted to meters
    calibrated: true
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCameraZones, setShowCameraZones] = useState(true);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(false);
  const [useQuadrangles, setUseQuadrangles] = useState(true);
  const [selectedObject, setSelectedObject] = useState<TrackedObject | null>(null);
  const [searchWarpId, setSearchWarpId] = useState(externalSearchQuery ?? '');
  const [highlightedObject, setHighlightedObject] = useState<number | null>(null);
  const [systemStatus, setSystemStatus] = useState({
    backend: 'unknown' as 'unknown' | 'connected' | 'disconnected',
    db: 'unknown' as 'unknown' | 'connected' | 'disconnected',
    cameras: { connected: 0, total: 0 }
  });

  const API_BASE = (import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000');

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/`);
      if (res.ok) {
        const d = await res.json();
        setSystemStatus(prev => ({
          ...prev,
          backend: 'connected',
          db: d.database === 'connected' ? 'connected' : 'disconnected'
        }));
      } else {
        setSystemStatus(prev => ({ ...prev, backend: 'disconnected', db: 'disconnected' }));
      }
    } catch (e) {
      setSystemStatus(prev => ({ ...prev, backend: 'disconnected', db: 'disconnected' }));
    }
  };

  const fetchCameraStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/system/multi-camera/status`);
      if (res.ok) {
        const d = await res.json();
        const connected = (d.active_cameras || 0) + (d.ready_cameras || 0);
        const total = d.total_cameras || 0;
        setSystemStatus(prev => ({ ...prev, cameras: { connected, total } }));
      }
    } catch (e) {
      // ignore
    }
  };


  // Poll system status periodically
  useEffect(() => {
    fetchSystemStatus();
    fetchCameraStatus();
    const t = setInterval(() => { fetchSystemStatus(); fetchCameraStatus(); }, 5000);
    return () => clearInterval(t);
  }, []);

  // Camera zones - UPDATED to match calibration files
  const cameraZones = [
    // Column 3 (Left side of screen) - Cameras 8,9,10,11
    { id: 8, name: "Camera 8", x_start: 120, x_end: 180, y_start: 0, y_end: 25, active: true },
    { id: 9, name: "Camera 9", x_start: 120, x_end: 180, y_start: 25, y_end: 50, active: true },
    { id: 10, name: "Camera 10", x_start: 120, x_end: 180, y_start: 50, y_end: 75, active: true },
    { id: 11, name: "Camera 11", x_start: 120, x_end: 180, y_start: 75, y_end: 100, active: true },
    // Column 2 (Middle) - Cameras 5,6,7
    { id: 5, name: "Camera 5", x_start: 60, x_end: 120, y_start: 0, y_end: 25, active: true },
    { id: 6, name: "Camera 6", x_start: 60, x_end: 120, y_start: 25, y_end: 50, active: true },
    { id: 7, name: "Camera 7", x_start: 60, x_end: 120, y_start: 50, y_end: 75, active: true },
    // Column 1 (Right side of screen) - Cameras 1,2,3,4 - CORRECTED COORDINATES
    { id: 1, name: "Camera 1", x_start: 0, x_end: 60, y_start: 0, y_end: 25, active: true },
    { id: 2, name: "Camera 2", x_start: 0, x_end: 60, y_start: 25, y_end: 50, active: true },
    { id: 3, name: "Camera 3", x_start: 0, x_end: 60, y_start: 50, y_end: 75, active: true },
    { id: 4, name: "Camera 4", x_start: 0, x_end: 60, y_start: 75, y_end: 100, active: true },
  ];

  const fetchObjects = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/tracking/objects`);
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

  // Sync external search from navbar
  useEffect(() => {
    if (externalSearchQuery !== undefined) {
      setSearchWarpId(externalSearchQuery);
      const value = externalSearchQuery;
      if (value && value.trim()) {
        const q = value.trim().toLowerCase();
        const matchingObject = objects.find(obj => {
          const byWarp = obj.warp_id && obj.warp_id.toLowerCase().includes(q);
          const byPersistent = obj.persistent_id !== undefined && String(obj.persistent_id).includes(q);
          const byGlobal = obj.global_id !== undefined && String(obj.global_id).includes(q);
          return byWarp || byPersistent || byGlobal;
        });
        if (matchingObject) {
          setHighlightedObject(matchingObject.persistent_id);
          setSelectedObject(matchingObject);
        } else {
          setHighlightedObject(null);
        }
      } else {
        setHighlightedObject(null);
      }
    }
  }, [externalSearchQuery, objects]);

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

  // Use the original simple color system from database with search highlighting
  const getObjectColor = (object: TrackedObject) => {
    // Use color from DB when available
    if (object.color_rgb && Array.isArray(object.color_rgb) && object.color_rgb.length >= 3) {
      return `rgb(${object.color_rgb[0]}, ${object.color_rgb[1]}, ${object.color_rgb[2]})`;
    }
    if (object.color_hex) {
      return object.color_hex;
    }
    if (object.color_name && (object.color_confidence ?? 1) > 0.3) {
      const colorMap: Record<string, string> = {
        red: '#ff4444',
        orange: '#ff8800',
        yellow: '#ffdd00',
        green: '#44ff44',
        blue: '#4444ff',
        purple: '#8844ff',
        pink: '#ff44aa',
        brown: '#8b4513',
        black: '#333333',
        white: '#f0f0f0',
        gray: '#888888',
        grey: '#888888',
        dark: '#444444',
      };
      const c = colorMap[object.color_name.toLowerCase()];
      if (c) return c;
    }
    // Fallback color
    return '#d97706';
  };

  // Convert physical coordinates to screen percentage
  const physicalToScreen = (physicalCoord: number[], warehouseWidthFt: number, warehouseLengthFt: number) => {
    const [physX, physY] = physicalCoord;
    // FIXED: Flipped mapping so Camera 8 (120-180ft) appears on YOUR LEFT side
    const x = ((warehouseWidthFt - physX) / warehouseWidthFt) * 100; // Flipped mapping
    const y = (physY / warehouseLengthFt) * 100; // Y-axis correct


    return [x, y];
  };

  // Render object as quadrangle if 4 corners available, otherwise as center point
  const renderObject = (object: TrackedObject, warehouseWidthFt: number, warehouseLengthFt: number) => {
    const objectColor = getObjectColor(object);
    const elements = [];



    // Check if we should use quadrangle rendering
    if (useQuadrangles &&
        object.physical_corners?.length === 4 &&
        object.physical_corners.every(corner =>
          Array.isArray(corner) &&
          corner.length >= 2 &&
          corner[0] !== null &&
          corner[1] !== null &&
          !isNaN(corner[0]) &&
          !isNaN(corner[1])
        )) {

      // Convert all 4 physical corners to screen coordinates
      const screenCorners = object.physical_corners.map(corner =>
        physicalToScreen(corner, warehouseWidthFt, warehouseLengthFt)
      );

      // Create SVG polygon points string
      const points = screenCorners.map(corner => `${corner[0]},${corner[1]}`).join(' ');

      elements.push(
        <polygon
          key={`quad-${object.persistent_id}`}
          points={points}
          fill={objectColor}
          stroke={highlightedObject === object.persistent_id ? '#3b82f6' : '#ffffff'}
          strokeWidth={highlightedObject === object.persistent_id ? '0.6' : '0.2'}
          fillOpacity={highlightedObject === object.persistent_id ? '1' : '0.85'}
          strokeOpacity={highlightedObject === object.persistent_id ? '1' : '0.5'}
          filter={highlightedObject === object.persistent_id ? 'url(#glow)' : undefined}
          style={{ cursor: 'pointer', pointerEvents: 'auto' }}
          onClick={() => setSelectedObject(object)}
        />
      );


    } else {
      // No center point rendering
    }

    // Add bounding box if enabled and available
    if (showBoundingBoxes && object.bbox && object.real_center) {
      const [centerX, centerY] = physicalToScreen(object.real_center, warehouseWidthFt, warehouseLengthFt);

      // Estimate bounding box size in screen coordinates (simplified)
      const boxWidth = 10; // Fixed width for visualization
      const boxHeight = 8; // Fixed height for visualization

      elements.push(
        <rect
          key={`bbox-${object.persistent_id}`}
          x={centerX - boxWidth/2}
          y={centerY - boxHeight/2}
          width={boxWidth}
          height={boxHeight}
          fill="none"
          stroke="#ff6b6b"
          strokeWidth="1"
          strokeDasharray="3,3"
          opacity="0.8"
        />
      );
    }

    return <g key={`object-${object.persistent_id}`}>{elements}</g>;
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
          <h2 className="text-2xl font-bold text-white tracking-tight">Live Object Tracking</h2>
          <div className="flex items-center gap-3">
            {/* Backend connection */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full ring-1 shadow-sm ${systemStatus.backend === 'connected' ? 'bg-emerald-900/20 ring-emerald-700/30' : 'bg-red-900/10 ring-red-700/30'}`}>
              <div className={`w-2 h-2 rounded-full ${systemStatus.backend === 'connected' ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span className="text-sm text-gray-200 font-medium">Backend {systemStatus.backend === 'connected' ? 'Connected' : 'Offline'}</span>
            </div>
            {/* Database connection */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full ring-1 shadow-sm ${systemStatus.db === 'connected' ? 'bg-emerald-900/20 ring-emerald-700/30' : 'bg-red-900/10 ring-red-700/30'}`}>
              <div className={`w-2 h-2 rounded-full ${systemStatus.db === 'connected' ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span className="text-sm text-gray-200 font-medium">DB {systemStatus.db === 'connected' ? 'Connected' : 'Offline'}</span>
            </div>
            {/* Cameras summary */}
            <div className="px-3 py-1 rounded-full bg-gradient-to-r from-gray-800/60 to-gray-800/30 ring-1 ring-blue-500/20 shadow-sm">
              <span className="text-sm text-blue-300 font-semibold">Cameras {systemStatus.cameras.connected}/{systemStatus.cameras.total}</span>
            </div>
            {/* Feeds toggle switch */}
            <div className="pl-3 ml-1 border-l border-gray-700 flex items-center gap-2">
              <span className="text-xs text-gray-300 flex items-center gap-1">
                <Camera className="w-3 h-3 text-gray-300" />
                Live Camera Feed
              </span>
              <button
                type="button"
                aria-pressed={(feedsEnabled ?? false) ? true : false}
                onClick={() => onToggleFeeds && onToggleFeeds(!(feedsEnabled ?? false))}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  (feedsEnabled ?? false) ? 'bg-green-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-5 w-5 transform bg-white rounded-full shadow transition-transform ${
                    (feedsEnabled ?? false) ? 'translate-x-5' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
        <div className="flex gap-6 text-sm text-gray-400">
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
            {/* Inbound docks segmented per right-column camera zone; plus Office area below Camera 7 */}
            {(() => {
              const warehouseWidthFt = config.width_feet || (config.width_meters * 3.28084);
              const warehouseLengthFt = config.length_feet || (config.length_meters * 3.28084);
              const rectStyle = (xStart: number, xEnd: number, yStart: number, yEnd: number) => ({
                left: `${((warehouseWidthFt - xEnd) / warehouseWidthFt) * 100}%`,
                top: `${(yStart / warehouseLengthFt) * 100}%`,
                width: `${((xEnd - xStart) / warehouseWidthFt) * 100}%`,
                height: `${((yEnd - yStart) / warehouseLengthFt) * 100}%`
              } as React.CSSProperties);


              return (
                <>
                  {/* Office area below Camera 7 (x:60-120, y:75-100) */}
                  <div
                    className="absolute pointer-events-none"
                    style={{
                      ...rectStyle(60, 120, 75, 100),
                      zIndex: 3
                    }}
                    aria-label="Office Area"
                  >
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="px-3 py-1 rounded bg-gray-900/40 border border-gray-600/60 text-gray-200 text-xs font-semibold tracking-wide backdrop-blur-sm">
                        OFFICE AREA
                      </div>
                    </div>
                  </div>
                </>
              );
            })()}

          <div
            className="relative bg-gray-600 border-2 border-gray-500 rounded-lg shadow-inner"
            style={{
              width: 'clamp(760px, 70vw, 900px)',
              aspectRatio: `${(config.width_feet || config.width_meters * 3.28084)} / ${(config.length_feet || config.length_meters * 3.28084)}`
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


                </div>
              );
            })}

            {/* Enhanced object markers with quadrangle support */}
            <svg
              className="absolute inset-0 w-full h-full"
              style={{ zIndex: 10, pointerEvents: 'none' }}
              width="100%"
              height="100%"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <defs>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="0.5" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>



              {objects.map((object) => {
                const warehouseWidthFt = config.width_feet || config.width_meters * 3.28084;
                const warehouseLengthFt = config.length_feet || config.length_meters * 3.28084;

                return renderObject(object, warehouseWidthFt, warehouseLengthFt);
              })}
            </svg>

            {/* Object ID labels overlay */}
            {false && objects.map((object) => {
              // Check if real_center exists
              if (!object.real_center) {
                return null;
              }

              const warehouseWidthFt = config.width_feet || config.width_meters * 3.28084;
              const warehouseLengthFt = config.length_feet || config.length_meters * 3.28084;
              const [x, y] = physicalToScreen(object.real_center, warehouseWidthFt, warehouseLengthFt);

              return (
                <div
                  key={`label-${object.persistent_id}`}
                  className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group pointer-events-auto"
                  style={{ left: `${x}%`, top: `${y}%`, zIndex: 20 }}
                  onClick={() => setSelectedObject(object)}
                >
                  {/* ID label with modern styling and Warp ID indicator */}
                  <div className={`text-white text-xs px-2 py-0.5 rounded-full font-semibold border shadow-lg flex items-center gap-1 ${
                    highlightedObject === object.persistent_id
                      ? 'bg-blue-600 border-blue-400 animate-pulse'
                      : 'bg-gray-800 border-gray-600'
                  }`}>
                    {object.persistent_id}
                    {/* Warp ID status indicator */}
                    {object.warp_id ? (
                      <div className="w-1.5 h-1.5 bg-green-400 rounded-full" title="QR Code Linked"></div>
                    ) : (
                      <div className="w-1.5 h-1.5 bg-orange-400 rounded-full" title="QR Code Not Linked"></div>
                    )}
                  </div>

                  {/* Enhanced tooltip with quadrangle info */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-3 opacity-0 group-hover:opacity-100 transition-all duration-200 bg-gray-900 text-white text-xs rounded-lg px-3 py-2 whitespace-nowrap z-30 border border-gray-600 shadow-xl">
                    <div className="space-y-1">
                      <div className="flex justify-between gap-3">
                        <span className="text-gray-400">ID:</span>
                        <span className="font-semibold">{object.persistent_id}</span>
                      </div>
                      {object.shape_type && (
                        <div className="flex justify-between gap-3">
                          <span className="text-gray-400">Shape:</span>
                          <span className="text-purple-400">{object.shape_type}</span>
                        </div>
                      )}
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
                      <div className="flex justify-between gap-3">
                        <span className="text-gray-400">Warp ID:</span>
                        {object.warp_id ? (
                          <span className="text-green-400 font-mono">{object.warp_id}</span>
                        ) : (
                          <span className="text-orange-400">Not Linked</span>
                        )}
                      </div>
                      {object.area && (
                        <div className="flex justify-between gap-3">
                          <span className="text-gray-400">Area:</span>
                          <span className="text-yellow-400">{object.area.toLocaleString()}px</span>
                        </div>
                      )}
                      {object.color_name && (
                        <div className="flex justify-between gap-3">
                          <span className="text-gray-400">Color:</span>
                          <span style={{ color: object.color_hex || '#fff' }}>{object.color_name}</span>
                        </div>
                      )}
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


          <div className="absolute -bottom-8 right-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            ({config.width_feet || (config.width_meters * 3.28084).toFixed(0)}ft, {config.length_feet || (config.length_meters * 3.28084).toFixed(0)}ft)
          </div>
          <div className="absolute -top-8 left-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            ({config.width_feet || (config.width_meters * 3.28084).toFixed(0)}ft, 0)
          </div>
          <div className="absolute -top-8 right-0 text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded border border-gray-600">
            (0, 0) - Origin
          </div>
        </div>
      </div>

      {/* Object Details Sidebar */}
      {selectedObject && (
        <div className="fixed top-20 right-4 w-80 bg-gray-800 rounded-xl border border-gray-600 shadow-2xl z-50">
          <div className="p-4 border-b border-gray-600 flex justify-between items-center">
            <h3 className="text-lg font-semibold text-white">Object Details</h3>
            <button
              onClick={() => setSelectedObject(null)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>
          <div className="p-4 space-y-4">
            {/* Basic Information */}
            <div className="space-y-2">
              <h4 className="text-sm font-semibold text-gray-300 border-b border-gray-600 pb-1">Basic Information</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-gray-400">Object ID:</div>
                <div className="text-white font-semibold">{selectedObject.persistent_id}</div>

                <div className="text-gray-400">Warp ID:</div>
                <div className="flex items-center gap-2">
                  {selectedObject.warp_id ? (
                    <>
                      <span className="text-green-400 font-mono font-semibold bg-green-900/20 px-2 py-1 rounded text-xs">
                        {selectedObject.warp_id}
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



                {selectedObject.camera_id && (
                  <>
                    <div className="text-gray-400">Camera:</div>
                    <div className="text-green-400">{selectedObject.camera_id}</div>
                  </>
                )}

                <div className="text-gray-400">Shape:</div>
                <div className="text-purple-400">{selectedObject.shape_type || 'rectangle'}</div>

                <div className="text-gray-400">Confidence:</div>
                <div className="text-green-400">{(selectedObject.confidence * 100).toFixed(1)}%</div>



                <div className="text-gray-400">Times Seen:</div>
                <div className="text-cyan-400">{selectedObject.times_seen}</div>



                {/* Timeline Information */}
                {(selectedObject.first_seen || selectedObject.last_seen || selectedObject.warp_id_linked_at) && (
                  <>
                    <div className="col-span-2 border-t border-gray-600 pt-2 mt-2">
                      <div className="text-xs font-semibold text-gray-300 mb-2">Timeline</div>
                    </div>

                    {selectedObject.first_seen && (
                      <>
                        <div className="text-gray-400">Inbound Time:</div>
                        <div className="text-blue-400 text-xs">{new Date(selectedObject.first_seen).toLocaleString()}</div>
                      </>
                    )}

                    {selectedObject.last_seen && (
                      <>
                        <div className="text-gray-400">Last Seen:</div>
                        <div className="text-green-400 text-xs">{new Date(selectedObject.last_seen).toLocaleString()}</div>
                      </>
                    )}

                    {selectedObject.warp_id_linked_at && (
                      <>
                        <div className="text-gray-400">Warp Linked:</div>
                        <div className="text-purple-400 text-xs">{new Date(selectedObject.warp_id_linked_at).toLocaleString()}</div>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>



            {/* Coordinate Information */}
            <div className="space-y-2">
              <h4 className="text-sm font-semibold text-gray-300 border-b border-gray-600 pb-1">Coordinates</h4>

              {/* Physical Center */}
              {selectedObject.real_center && (
                <div className="bg-gray-700 p-3 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Physical Center</div>
                  <div className="text-blue-400 font-mono">
                    ({selectedObject.real_center[0]?.toFixed(2)}ft, {selectedObject.real_center[1]?.toFixed(2)}ft)
                  </div>
                </div>
              )}



              {/* Physical Corners */}
              {selectedObject.physical_corners && selectedObject.physical_corners.length === 4 && (
                <div className="bg-gray-700 p-3 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">Physical Corners (feet)</div>
                  <div className="text-purple-400 font-mono text-xs space-y-1">
                    {selectedObject.physical_corners.map((corner, i) => (
                      <div key={i}>
                        Corner {i+1}: ({corner[0]?.toFixed(2)}ft, {corner[1]?.toFixed(2)}ft)
                      </div>
                    ))}
                  </div>
                </div>
              )}


            </div>

            {/* Recent Crops Gallery */}
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-sm text-gray-400 mb-2">🖼️ Recent Crops</div>
              <ObjectCropsGallery persistentId={selectedObject.persistent_id || selectedObject.global_id as any} />
            </div>



            {/* Action */}
            <div className="pt-2">
              <button
                onClick={() => {
                  // Placeholder hook to send data to robot
                  // TODO: wire actual API call when endpoint is ready
                  console.log('Send to robot:', selectedObject);
                }}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
              >
                Send data to robot
              </button>
            </div>
          </div>
        </div>
      )}


    </div>
  );
};

export default WorkingWarehouseView;
