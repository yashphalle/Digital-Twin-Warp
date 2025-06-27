import React, { useState } from 'react';
import { useTracking, useWarehouseConfig } from '../hooks/useTracking';
import ObjectMarker from './ObjectMarker';

interface CameraZone {
  camera_id: number;
  camera_name: string;
  x_start: number;
  x_end: number;
  y_start: number;
  y_end: number;
  active: boolean;
}

const MultiCameraWarehouseView: React.FC = () => {
  const { objects, loading: objectsLoading, error: objectsError } = useTracking();
  const { config, loading: configLoading, error: configError } = useWarehouseConfig();
  
  const [showCameraZones, setShowCameraZones] = useState(true);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);

  // Multi-camera warehouse config - Use API config if available, otherwise fallback
  const warehouseConfig = {
    width_ft: config?.width_feet || 180.0,
    length_ft: config?.length_feet || 90.0,
    width_meters: config?.width_meters || (180.0 * 0.3048),
    length_meters: config?.length_meters || (90.0 * 0.3048),
    units: config?.units || 'feet'
  };

  // Camera zones based on the 11-camera layout (4-3-4 arrangement)
  const cameraZones: CameraZone[] = [
    // Row 1 (Front) - 4 cameras
    { camera_id: 1, camera_name: "Camera 1 - Front Left", x_start: 0, x_end: 45, y_start: 0, y_end: 30, active: false },
    { camera_id: 2, camera_name: "Camera 2 - Front Center-Left", x_start: 35, x_end: 80, y_start: 0, y_end: 30, active: false },
    { camera_id: 3, camera_name: "Camera 3 - Front Center-Right", x_start: 70, x_end: 115, y_start: 0, y_end: 30, active: false },
    { camera_id: 4, camera_name: "Camera 4 - Front Right", x_start: 105, x_end: 150, y_start: 0, y_end: 30, active: false },
    
    // Row 2 (Middle) - 3 cameras
    { camera_id: 5, camera_name: "Camera 5 - Middle Left", x_start: 0, x_end: 60, y_start: 25, y_end: 55, active: false },
    { camera_id: 6, camera_name: "Camera 6 - Middle Center", x_start: 45, x_end: 105, y_start: 25, y_end: 55, active: false },
    { camera_id: 7, camera_name: "Camera 7 - Middle Right", x_start: 90, x_end: 150, y_start: 25, y_end: 55, active: false },
    
    // Column 3 (Left) - 4 cameras - PHASE 1: All Column 3 cameras active
    { camera_id: 8, camera_name: "Camera 8 - Column 3 Top", x_start: 120, x_end: 180, y_start: 0, y_end: 25, active: true },
    { camera_id: 9, camera_name: "Camera 9 - Column 3 Mid-Top", x_start: 120, x_end: 180, y_start: 25, y_end: 50, active: true },
    { camera_id: 10, camera_name: "Camera 10 - Column 3 Mid-Bottom", x_start: 120, x_end: 180, y_start: 50, y_end: 75, active: true },
    { camera_id: 11, camera_name: "Camera 11 - Column 3 Bottom", x_start: 120, x_end: 180, y_start: 75, y_end: 90, active: true }
  ];

  if (configLoading || objectsLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-lg text-gray-600">Loading multi-camera warehouse data...</div>
      </div>
    );
  }

  // Convert feet to pixels for visualization (scale warehouse to fit screen)
  const pixelsPerFoot = 8; // 8 pixels per foot for better visibility
  const warehousePixelWidth = warehouseConfig.width_ft * pixelsPerFoot; // 1440px
  const warehousePixelHeight = warehouseConfig.length_ft * pixelsPerFoot; // 720px

  // Convert object coordinates (assuming they're already in feet)
  const convertToPixels = (object: any) => {
    // FIXED: Flipped mapping so Camera 8 (120-180ft) appears on YOUR LEFT side
    const x = (warehouseConfig.width_ft - (object.real_center?.[0] || 0)) * pixelsPerFoot; // Flipped mapping
    const y = (object.real_center?.[1] || 0) * pixelsPerFoot; // Y-axis correct
    return { x, y };
  };

  const getCameraZoneStyle = (zone: CameraZone) => {
    // FIXED: Flipped mapping so Camera 8 (120-180ft) appears on YOUR LEFT side
    const x = (warehouseConfig.width_ft - zone.x_end) * pixelsPerFoot; // Flipped mapping
    const y = zone.y_start * pixelsPerFoot; // Y-axis correct
    const width = (zone.x_end - zone.x_start) * pixelsPerFoot;
    const height = (zone.y_end - zone.y_start) * pixelsPerFoot;

    return {
      position: 'absolute' as const,
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      border: zone.active ? '3px solid #10b981' : '2px solid #6b7280',
      backgroundColor: zone.active ? 'rgba(16, 185, 129, 0.1)' : 'rgba(107, 114, 128, 0.05)',
      cursor: 'pointer',
      transition: 'all 0.2s ease-in-out'
    };
  };

  const activeCameras = cameraZones.filter(zone => zone.active);
  const readyCameras = cameraZones.filter(zone => !zone.active);

  return (
    <div className="multi-camera-warehouse-container p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Multi-Camera Warehouse Tracking</h2>
        <div className="flex gap-6 text-sm text-gray-600 mb-4">
          <span>Objects: {objects?.length || 0}</span>
          <span>Warehouse: {warehouseConfig.width_ft}ft × {warehouseConfig.length_ft}ft</span>
          <span className="text-green-600">✓ 11-Camera System</span>
          <span className="text-green-600">Active: Cameras 8, 9, 10, 11 (Column 3)</span>
          <span className="text-blue-600">📹 Camera zones visible</span>
          {(objectsError || configError) && (
            <span className="text-red-600">⚠ {objectsError || configError}</span>
          )}
        </div>
        
        {/* Controls */}
        <div className="flex gap-4 items-center flex-wrap">
          <button
            onClick={() => setShowCameraZones(!showCameraZones)}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              showCameraZones
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {showCameraZones ? 'Hide' : 'Show'} Camera Zones
          </button>

          <div className="text-sm text-gray-500">
            Click camera zones to view details
          </div>

          {/* Camera zone legend */}
          {showCameraZones && (
            <div className="flex gap-4 items-center text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-green-500 rounded border"></div>
                <span>Active Camera</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-gray-400 rounded border"></div>
                <span>Standby Camera</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Feed</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Warehouse visualization */}
      <div className="warehouse-wrapper flex justify-center items-start">
        <div className="relative">
          {/* Dimension labels */}
          <div
            className="absolute text-sm text-gray-600 font-medium"
            style={{
              top: '-25px',
              left: '50%',
              transform: 'translateX(-50%)',
              whiteSpace: 'nowrap'
            }}
          >
            Width: {warehouseConfig.width_ft}ft ({warehouseConfig.width_meters.toFixed(1)}m)
          </div>
          
          <div
            className="absolute text-sm text-gray-600 font-medium"
            style={{
              left: '-100px',
              top: '50%',
              transform: 'translateY(-50%) rotate(-90deg)',
              whiteSpace: 'nowrap'
            }}
          >
            Length: {warehouseConfig.length_ft}ft ({warehouseConfig.length_meters.toFixed(1)}m)
          </div>

          {/* Warehouse boundary */}
          <div
            className="warehouse-boundary relative bg-gray-50 border-2 border-gray-400 overflow-hidden"
            style={{
              width: `${warehousePixelWidth}px`,
              height: `${warehousePixelHeight}px`
            }}
          >
            {/* Grid lines every 30 feet */}
            <div className="absolute inset-0 opacity-20">
              {/* Vertical lines every 30ft */}
              {Array.from({ length: Math.floor(warehouseConfig.width_ft / 30) + 1 }, (_, i) => (
                <div
                  key={`v-${i}`}
                  className="absolute h-full border-l border-gray-400"
                  style={{ left: `${i * 30 * pixelsPerFoot}px` }}
                />
              ))}
              {/* Horizontal lines every 30ft */}
              {Array.from({ length: Math.floor(warehouseConfig.length_ft / 30) + 1 }, (_, i) => (
                <div
                  key={`h-${i}`}
                  className="absolute w-full border-t border-gray-400"
                  style={{ top: `${i * 30 * pixelsPerFoot}px` }}
                />
              ))}
            </div>

            {/* Camera zones */}
            {showCameraZones && cameraZones.map((zone) => (
              <div
                key={zone.camera_id}
                style={getCameraZoneStyle(zone)}
                onClick={() => setSelectedCamera(selectedCamera === zone.camera_id ? null : zone.camera_id)}
                className={`absolute border-2 cursor-pointer transition-all duration-200 ${
                  zone.active
                    ? 'border-green-500 bg-green-100 bg-opacity-40 hover:bg-opacity-60'
                    : 'border-gray-400 bg-gray-200 bg-opacity-30 hover:bg-opacity-50'
                } ${selectedCamera === zone.camera_id ? 'ring-4 ring-blue-300' : ''}`}
                title={`${zone.camera_name} - Click for details`}
              >
                {/* Camera ID badge */}
                <div className={`absolute top-2 left-2 text-sm font-bold px-2 py-1 rounded-full shadow-sm ${
                  zone.active
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-600 text-white'
                }`}>
                  {zone.camera_id}
                </div>

                {/* Active indicator */}
                {zone.active && (
                  <div className="absolute top-2 right-2 w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-sm">
                    <div className="absolute inset-0 bg-green-400 rounded-full animate-ping"></div>
                  </div>
                )}

                {/* Camera name */}
                <div className="absolute bottom-2 left-2 text-xs bg-black bg-opacity-75 text-white px-2 py-1 rounded shadow-sm max-w-full truncate">
                  {zone.camera_name.replace('Camera ', 'Cam ').replace(' - Column', ' Col')}
                </div>

                {/* Coverage coordinates */}
                <div className="absolute bottom-2 right-2 text-xs bg-black bg-opacity-75 text-white px-2 py-1 rounded shadow-sm">
                  {zone.x_start}-{zone.x_end}×{zone.y_start}-{zone.y_end}ft
                </div>

                {/* Center crosshair for active cameras */}
                {zone.active && (
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="w-4 h-4 border-2 border-green-600 rounded-full bg-white bg-opacity-80">
                      <div className="w-full h-full flex items-center justify-center">
                        <div className="w-1 h-1 bg-green-600 rounded-full"></div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Selected camera details overlay */}
                {selectedCamera === zone.camera_id && (
                  <div className="absolute inset-0 bg-blue-500 bg-opacity-20 border-2 border-blue-500 rounded flex items-center justify-center">
                    <div className="bg-white bg-opacity-95 p-3 rounded shadow-lg text-xs max-w-full">
                      <div className="font-bold text-gray-800">{zone.camera_name}</div>
                      <div className="text-gray-600 mt-1">
                        Coverage: {zone.x_start}-{zone.x_end}ft × {zone.y_start}-{zone.y_end}ft
                      </div>
                      <div className="text-gray-600">
                        Size: {zone.x_end - zone.x_start}ft × {zone.y_end - zone.y_start}ft
                      </div>
                      <div className={`mt-1 font-medium ${zone.active ? 'text-green-600' : 'text-gray-500'}`}>
                        Status: {zone.active ? '🟢 Active' : '⚪ Standby'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Object markers - TEST BOXES */}
            {objects && objects.map((object) => {
              const pixelPos = convertToPixels(object);
              
              console.log('=== OBJECT DEBUG ===');
              console.log('Object:', object);
              console.log('Pixel position:', pixelPos);
              console.log('Pixels per foot:', pixelsPerFoot);
              
              return (
                <div
                  key={object.persistent_id}
                  style={{
                    position: 'absolute',
                    left: `${pixelPos.x}px`,
                    top: `${pixelPos.y}px`,
                    width: '100px',
                    height: '80px',
                    backgroundColor: '#ff0000',
                    border: '4px solid #000000',
                    zIndex: 10,
                    transform: 'translate(-50%, -50%)'
                  }}
                >
                  <div style={{
                    position: 'absolute',
                    top: '-25px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    backgroundColor: '#000000',
                    color: '#ffffff',
                    padding: '2px 6px',
                    fontSize: '12px',
                    borderRadius: '3px'
                  }}>
                    ID: {object.persistent_id}
                  </div>
                  
                  <div style={{
                    position: 'absolute',
                    top: '2px',
                    left: '2px',
                    backgroundColor: '#000000',
                    color: '#ffffff',
                    padding: '1px 3px',
                    fontSize: '10px'
                  }}>
                    100×80px
                  </div>
                </div>
              );
            })}
            
            {/* TEST BOX - Always visible regardless of data */}
            <div
              style={{
                position: 'absolute',
                left: '200px',
                top: '100px',
                width: '120px',
                height: '80px',
                backgroundColor: '#00ff00',
                border: '4px solid #000000',
                zIndex: 15
              }}
            >
              <div style={{
                position: 'absolute',
                top: '-25px',
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: '#000000',
                color: '#ffffff',
                padding: '2px 6px',
                fontSize: '12px',
                borderRadius: '3px'
              }}>
                TEST BOX
              </div>
            </div>

            {/* Empty state */}
            {(!objects || objects.length === 0) && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-gray-500 text-center bg-white/80 p-4 rounded">
                  <div className="text-lg mb-2">
                    No objects detected
                  </div>
                  <div className="text-sm">
                    Objects from Camera 8 will appear here when detected
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Coordinate indicators - NEW COORDINATE SYSTEM (Origin: Top-Right) */}
          <div className="absolute -top-4 right-0 text-xs text-gray-500 font-semibold">
            (0ft, 0ft) ← ORIGIN
          </div>
          <div className="absolute -top-4 left-0 text-xs text-gray-500">
            ({warehouseConfig.width_ft}ft, 0ft)
          </div>
          <div className="absolute -bottom-6 right-0 text-xs text-gray-500">
            (0ft, {warehouseConfig.length_ft}ft)
          </div>
          <div className="absolute -bottom-6 left-0 text-xs text-gray-500">
            ({warehouseConfig.width_ft}ft, {warehouseConfig.length_ft}ft)
          </div>
        </div>
      </div>

      {/* Camera Information Panel */}
      {selectedCamera && (
        <div className="mt-6 bg-white p-4 rounded-lg shadow border">
          {(() => {
            const zone = cameraZones.find(z => z.camera_id === selectedCamera);
            return zone ? (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">{zone.camera_name}</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Status:</span> 
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      zone.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}>
                      {zone.active ? 'ACTIVE' : 'READY'}
                    </span>
                  </div>
                  <div>
                    <span className="font-medium">Coverage Area:</span> {zone.x_start}-{zone.x_end}ft × {zone.y_start}-{zone.y_end}ft
                  </div>
                  <div>
                    <span className="font-medium">Size:</span> {zone.x_end - zone.x_start}ft × {zone.y_end - zone.y_start}ft
                  </div>
                  <div>
                    <span className="font-medium">Position:</span> Row {Math.floor((selectedCamera - 1) / 5) + 1}
                  </div>
                </div>
                {zone.active && (
                  <div className="mt-2 text-sm text-green-600">
                    ✓ Currently processing objects and sending data to warehouse system
                  </div>
                )}
              </div>
            ) : null;
          })()}
        </div>
      )}

      {/* System Status */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-white p-4 rounded-lg shadow border text-center">
          <div className="text-2xl font-bold text-green-600">{activeCameras.length}</div>
          <div className="text-sm text-gray-600">Active Cameras</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border text-center">
          <div className="text-2xl font-bold text-gray-600">{readyCameras.length}</div>
          <div className="text-sm text-gray-600">Ready Cameras</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border text-center">
          <div className="text-2xl font-bold text-blue-600">{objects?.length || 0}</div>
          <div className="text-sm text-gray-600">Tracked Objects</div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 flex justify-center">
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Legend</h3>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-green-500 bg-green-100"></div>
                <span>Active Camera Zone</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-gray-500 bg-gray-100"></div>
                <span>Ready Camera Zone</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-8 h-6 border-2 border-orange-500 bg-orange-200/40 relative">
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                    <div className="w-0.5 h-2 bg-orange-600"></div>
                    <div className="w-2 h-0.5 bg-orange-600 absolute top-0.5 -left-0.5"></div>
                  </div>
                </div>
                <span>Pallet (≥4×4ft) 📦</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-4 border-2 border-red-500 bg-red-200/40 relative">
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                    <div className="w-0.5 h-2 bg-red-600"></div>
                    <div className="w-2 h-0.5 bg-red-600 absolute top-0.5 -left-0.5"></div>
                  </div>
                </div>
                <span>Package (&lt;4×4ft) 📋</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border border-gray-400 bg-gray-50"></div>
                <span>30ft Grid</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MultiCameraWarehouseView; 