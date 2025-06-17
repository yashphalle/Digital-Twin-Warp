import React, { useState } from 'react';
import { TrackedObject, WarehouseConfig } from '../types/tracking';

interface ObjectMarkerProps {
  object: TrackedObject;
  warehouseConfig: WarehouseConfig;
}

const ObjectMarker: React.FC<ObjectMarkerProps> = ({ object, warehouseConfig }) => {
  const [showDetails, setShowDetails] = useState(false);

  // Convert real coordinates to pixel position (percentage)
  const getPosition = () => {
    if (object.real_center) {
      const pixelX = (object.real_center.x / warehouseConfig.width_meters) * 100;
      const pixelY = (object.real_center.y / warehouseConfig.length_meters) * 100;
      return { x: pixelX, y: pixelY };
    }
    // Fallback to center coordinates if real_center not available
    // Assuming center coordinates are normalized to warehouse dimensions
    return { 
      x: (object.center.x / 1000) * 100, // Adjust based on actual coordinate system
      y: (object.center.y / 1000) * 100 
    };
  };

  const position = getPosition();

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
        zIndex: 10
      }}
      onClick={() => setShowDetails(!showDetails)}
      onMouseEnter={() => setShowDetails(true)}
      onMouseLeave={() => setShowDetails(false)}
    >
      {/* Marker dot */}
      <div
        className="marker-dot"
        style={{
          width: '12px',
          height: '12px',
          borderRadius: '50%',
          backgroundColor: statusInfo.color,
          border: '2px solid white',
          boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
          transition: 'all 0.2s ease'
        }}
      />
      
      {/* Object ID label */}
      <div
        className="marker-label"
        style={{
          fontSize: '10px',
          textAlign: 'center',
          marginTop: '2px',
          whiteSpace: 'nowrap',
          color: '#333',
          fontWeight: 'bold',
          textShadow: '1px 1px 2px rgba(255,255,255,0.8)'
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
            zIndex: 20
          }}
        >
          <div><strong>ID:</strong> {object.persistent_id}</div>
          <div><strong>Status:</strong> {statusInfo.status}</div>
          <div><strong>Age:</strong> {formatAge(object.age_seconds)}</div>
          <div><strong>Confidence:</strong> {(object.confidence * 100).toFixed(1)}%</div>
          <div><strong>Times Seen:</strong> {object.times_seen}</div>
          {object.real_center && (
            <div><strong>Position:</strong> ({object.real_center.x.toFixed(1)}m, {object.real_center.y.toFixed(1)}m)</div>
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
