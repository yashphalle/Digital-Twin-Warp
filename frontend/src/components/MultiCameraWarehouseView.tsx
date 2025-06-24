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
    
    // Row 3 (Back) - 4 cameras
    { camera_id: 8, camera_name: "Camera 8 - Back Left", x_start: 0, x_end: 45, y_start: 50, y_end: 80, active: true }, // Currently active
    { camera_id: 9, camera_name: "Camera 9 - Back Center-Left", x_start: 35, x_end: 80, y_start: 50, y_end: 80, active: false },
    { camera_id: 10, camera_name: "Camera 10 - Back Center-Right", x_start: 70, x_end: 115, y_start: 50, y_end: 80, active: false },
    { camera_id: 11, camera_name: "Camera 11 - Back Right", x_start: 105, x_end: 150, y_start: 50, y_end: 80, active: false }
  ];

  if (configLoading || objectsLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-lg text-gray-600">Loading multi-camera warehouse data...</div>
      </div>
    );
  }

  // Convert feet to pixels for visualization (scale warehouse to fit screen)
  const pixelsPerFoot = 4; // 4 pixels per foot
  const warehousePixelWidth = warehouseConfig.width_ft * pixelsPerFoot; // 720px
  const warehousePixelHeight = warehouseConfig.length_ft * pixelsPerFoot; // 360px

  // Convert object coordinates (assuming they're already in feet)
  const convertToPixels = (object: any) => {
    const x = (object.real_center_x || 0) * pixelsPerFoot;
    const y = (object.real_center_y || 0) * pixelsPerFoot;
    return { x, y };
  };

  const getCameraZoneStyle = (zone: CameraZone) => {
    const x = zone.x_start * pixelsPerFoot;
    const y = zone.y_start * pixelsPerFoot;
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
          <span className="text-blue-600">Active: Camera 8</span>
        </div>
        
        {/* Controls */}
        <div className="flex gap-4 items-center">
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
                className="hover:opacity-80"
                title={zone.camera_name}
              >
                {/* Camera label */}
                <div className={`absolute top-1 left-1 text-xs font-semibold px-2 py-1 rounded ${
                  zone.active 
                    ? 'bg-green-500 text-white' 
                    : 'bg-gray-500 text-white'
                }`}>
                  C{zone.camera_id}
                </div>
                
                {/* Coverage area indicator */}
                <div className="absolute bottom-1 right-1 text-xs text-gray-600">
                  {zone.x_end - zone.x_start}×{zone.y_end - zone.y_start}ft
                </div>
              </div>
            ))}

            {/* Object markers */}
            {objects && objects.map((object) => {
              const pixelPos = convertToPixels(object);
              return (
                <div
                  key={object.persistent_id}
                  className="absolute w-3 h-3 bg-red-500 rounded-full border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2"
                  style={{
                    left: `${pixelPos.x}px`,
                    top: `${pixelPos.y}px`,
                    zIndex: 10
                  }}
                  title={`Object ${object.persistent_id} - (${object.real_center_x?.toFixed(1) || 0}, ${object.real_center_y?.toFixed(1) || 0})ft`}
                />
              );
            })}

            {/* Empty state */}
            {(!objects || objects.length === 0) && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-gray-500 text-center bg-white/80 p-4 rounded">
                  <div className="text-lg mb-2">
                    No objects detected
                  </div>
                  <div className="text-sm">
                    Objects from Camera 7 will appear here when detected
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Coordinate indicators */}
          <div className="absolute -bottom-6 left-0 text-xs text-gray-500">
            (0ft, 0ft)
          </div>
          <div className="absolute -bottom-6 right-0 text-xs text-gray-500">
            ({warehouseConfig.width_ft}ft, 0ft)
          </div>
          <div className="absolute -top-4 left-0 text-xs text-gray-500">
            (0ft, {warehouseConfig.length_ft}ft)
          </div>
          <div className="absolute -top-4 right-0 text-xs text-gray-500">
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
                <div className="w-3 h-3 rounded-full bg-red-500 border-2 border-white"></div>
                <span>Tracked Object</span>
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