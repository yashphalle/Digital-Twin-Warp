import React, { useState } from 'react';
import type { TrackedObject, WarehouseConfig } from '../types/tracking';

interface ObjectMarkerProps {
  object: TrackedObject;
  warehouseConfig: WarehouseConfig;
}

const ObjectMarker: React.FC<ObjectMarkerProps> = ({ object, warehouseConfig }) => {
  const [showDetails, setShowDetails] = useState(false);

  // Convert real coordinates (in feet) to pixel position (percentage)
  const getPosition = () => {
    if (object.real_center && Array.isArray(object.real_center) && object.real_center.length >= 2) {
      const warehouseWidthFt = warehouseConfig.width_feet || warehouseConfig.width_meters * 3.28084;
      const warehouseLengthFt = warehouseConfig.length_feet || warehouseConfig.length_meters * 3.28084;

      // FIXED: Flipped mapping so Camera 8 (120-180ft) appears on YOUR LEFT side
      const pixelX = ((warehouseWidthFt - object.real_center[0]) / warehouseWidthFt) * 100; // Flipped mapping
      const pixelY = (object.real_center[1] / warehouseLengthFt) * 100; // Y-axis correct
      return { x: pixelX, y: pixelY };
    }
    // Fallback to center coordinates if real_center not available
    // Assuming center coordinates are normalized to warehouse dimensions
    return { 
      x: (object.center.x / 1000) * 100, // Adjust based on actual coordinate system
      y: (object.center.y / 1000) * 100 
    };
  };

  // Calculate real-world size of object from bounding box
  const getObjectSize = () => {
    if (!object.bbox || !object.real_center) {
      // Default size if no bbox info - assume typical person size
      return { width: 2.0, height: 6.0 }; // 2ft x 6ft for person
    }

    // Calculate pixel size from bbox
    const [xmin, ymin, xmax, ymax] = object.bbox;
    const pixelWidth = xmax - xmin;
    const pixelHeight = ymax - ymin;

    // Estimate real-world size based on typical camera calibration
    // This is an approximation - for accurate sizing, we'd need camera intrinsics
    // Assuming average detection at medium distance gives ~30-50 pixels per foot
    const pixelsPerFoot = 40; // Rough approximation
    
    const realWidth = Math.max(pixelWidth / pixelsPerFoot, 1.0); // Min 1ft width
    const realHeight = Math.max(pixelHeight / pixelsPerFoot, 2.0); // Min 2ft height

    // Clamp to reasonable object sizes (person: 1-4ft wide, 4-7ft tall)
    return {
      width: Math.min(Math.max(realWidth, 1.0), 4.0),
      height: Math.min(Math.max(realHeight, 4.0), 7.0)
    };
  };

  // Convert object size to display size (percentage of warehouse)
  const getDisplaySize = () => {
    const objectSize = getObjectSize();
    const warehouseWidthFt = warehouseConfig.width_feet || warehouseConfig.width_meters * 3.28084;
    const warehouseLengthFt = warehouseConfig.length_feet || warehouseConfig.length_meters * 3.28084;
    
    const widthPercent = (objectSize.width / warehouseWidthFt) * 100;
    const heightPercent = (objectSize.height / warehouseLengthFt) * 100;
    
    return { 
      width: Math.max(widthPercent, 0.3), // Minimum 0.3% for visibility
      height: Math.max(heightPercent, 0.8)  // Minimum 0.8% for visibility
    };
  };

  const position = getPosition();
  const displaySize = getDisplaySize();

  // Determine object status and color
  const getStatusInfo = () => {
    if (object.status) {
      return {
        status: object.status,
        color: getStatusColor(object.status)
      };
    }
    
    // Determine status based on age if not provided
    if (object.age_seconds < 5) {
      return { status: 'new', color: '#ffd93d' };
    } else if (object.age_seconds < 60) {
      return { status: 'tracking', color: '#6bcf7f' };
    } else {
      return { status: 'established', color: '#4ecdc4' };
    }
  };

  const statusInfo = getStatusInfo();

  const formatAge = (seconds: number) => {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
      return `${Math.round(seconds / 60)}m`;
    } else {
      return `${Math.round(seconds / 3600)}h`;
    }
  };

  return (
    <div
      className="object-marker"
      style={{
        left: `${position.x}%`,
        top: `${position.y}%`,
        position: 'absolute',
        transform: 'translate(-50%, -50%)',
        cursor: 'pointer',
        zIndex: 10,
        width: `${displaySize.width}%`,
        height: `${displaySize.height}%`
      }}
      onClick={() => setShowDetails(!showDetails)}
      onMouseEnter={() => setShowDetails(true)}
      onMouseLeave={() => setShowDetails(false)}
    >
      {/* Object box */}
      <div
        className="object-box"
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: statusInfo.color,
          border: '2px solid white',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
          transition: 'all 0.2s ease',
          opacity: showDetails ? 0.8 : 0.6
        }}
      />
      
      {/* Center point indicator */}
      <div
        className="center-dot"
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '4px',
          height: '4px',
          borderRadius: '50%',
          backgroundColor: 'white',
          border: '1px solid black',
          zIndex: 1
        }}
      />
      
      {/* Object ID label */}
      <div
        className="marker-label"
        style={{
          position: 'absolute',
          top: '-20px',
          left: '50%',
          transform: 'translateX(-50%)',
          fontSize: '10px',
          textAlign: 'center',
          whiteSpace: 'nowrap',
          color: '#333',
          fontWeight: 'bold',
          textShadow: '1px 1px 2px rgba(255,255,255,0.8)',
          backgroundColor: 'rgba(255,255,255,0.8)',
          padding: '1px 4px',
          borderRadius: '2px'
        }}
      >
        ID: {object.persistent_id}
      </div>

      {/* Details tooltip */}
      {showDetails && (
        <div
          className="marker-details"
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            whiteSpace: 'nowrap',
            marginBottom: '5px',
            boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
            zIndex: 20,
            minWidth: '200px'
          }}
        >
          <div><strong>ID:</strong> {object.persistent_id}</div>
          <div><strong>Status:</strong> {statusInfo.status}</div>
          <div><strong>Age:</strong> {formatAge(object.age_seconds)}</div>
          <div><strong>Confidence:</strong> {(object.confidence * 100).toFixed(1)}%</div>
          <div><strong>Times Seen:</strong> {object.times_seen}</div>
          {object.real_center && Array.isArray(object.real_center) && object.real_center.length >= 2 && (
            <div><strong>Position:</strong> ({object.real_center[0].toFixed(1)}ft, {object.real_center[1].toFixed(1)}ft)</div>
          )}
          {object.bbox && (
            <div><strong>Size:</strong> {getObjectSize().width.toFixed(1)}ft × {getObjectSize().height.toFixed(1)}ft</div>
          )}
          
          {/* Arrow pointing down */}
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '5px solid transparent',
              borderRight: '5px solid transparent',
              borderTop: '5px solid rgba(0, 0, 0, 0.9)'
            }}
          />
        </div>
      )}
    </div>
  );
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'new':
      return '#ffd93d'; // Yellow
    case 'tracking':
      return '#6bcf7f'; // Green
    case 'established':
      return '#4ecdc4'; // Teal
    default:
      return '#ff6b6b'; // Red (fallback)
  }
};

export default ObjectMarker;
