import React from 'react';
import { useTracking, useWarehouseConfig } from '../hooks/useTracking';
import ObjectMarker from './ObjectMarker';

const WarehouseView: React.FC = () => {
  const { objects, loading: objectsLoading, error: objectsError } = useTracking();
  const { config, loading: configLoading, error: configError } = useWarehouseConfig();

  // Fallback config for testing
  const fallbackConfig = {
    width_meters: 10.0,
    length_meters: 8.0,
    calibrated: true,
    last_updated: new Date().toISOString()
  };

  // Use fallback config if there's an error or no config
  const warehouseConfig = config || fallbackConfig;

  if (configLoading || objectsLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-lg text-gray-600">Loading warehouse data...</div>
      </div>
    );
  }

  // Show error but continue with fallback data
  const hasError = configError || objectsError;

  return (
    <div className="warehouse-container p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Warehouse Tracking View</h2>
        <div className="flex gap-4 text-sm text-gray-600">
          <span>Objects: {objects?.length || 0}</span>
          <span>Dimensions: {warehouseConfig.width_meters}m × {warehouseConfig.length_meters}m</span>
          <span className={`${warehouseConfig.calibrated ? 'text-green-600' : 'text-red-600'}`}>
            {warehouseConfig.calibrated ? '✓ Calibrated' : '⚠ Not Calibrated'}
          </span>
          {hasError && (
            <span className="text-red-500 text-xs">⚠ Using fallback data</span>
          )}
        </div>
      </div>

      {/* Warehouse visualization */}
      <div className="warehouse-wrapper flex justify-center items-center" style={{ minHeight: '600px' }}>
        <div className="relative">
          {/* Dimension labels */}
          <div
            className="dimension-label-top absolute text-sm text-gray-600 font-medium"
            style={{
              top: '-25px',
              left: '50%',
              transform: 'translateX(-50%)',
              whiteSpace: 'nowrap'
            }}
          >
            Width: {config.width_meters}m
          </div>
          
          <div
            className="dimension-label-side absolute text-sm text-gray-600 font-medium"
            style={{
              left: '-80px',
              top: '50%',
              transform: 'translateY(-50%) rotate(-90deg)',
              whiteSpace: 'nowrap'
            }}
          >
            Length: {config.length_meters}m
          </div>

          {/* Warehouse boundary */}
          <div
            className="warehouse-boundary relative bg-gray-50 border-2 border-gray-400"
            style={{
              width: '600px',
              height: `${(warehouseConfig.length_meters / warehouseConfig.width_meters) * 600}px`,
              maxHeight: '500px',
              minHeight: '300px'
            }}
          >
            {/* Grid lines for reference (optional) */}
            <div className="absolute inset-0 opacity-20">
              {/* Vertical lines */}
              {Array.from({ length: Math.floor(warehouseConfig.width_meters) + 1 }, (_, i) => (
                <div
                  key={`v-${i}`}
                  className="absolute h-full border-l border-gray-300"
                  style={{ left: `${(i / warehouseConfig.width_meters) * 100}%` }}
                />
              ))}
              {/* Horizontal lines */}
              {Array.from({ length: Math.floor(warehouseConfig.length_meters) + 1 }, (_, i) => (
                <div
                  key={`h-${i}`}
                  className="absolute w-full border-t border-gray-300"
                  style={{ top: `${(i / warehouseConfig.length_meters) * 100}%` }}
                />
              ))}
            </div>

            {/* Object markers */}
            {objects && objects.map((object) => (
              <ObjectMarker
                key={object.persistent_id}
                object={object}
                warehouseConfig={warehouseConfig}
              />
            ))}

            {/* Empty state */}
            {(!objects || objects.length === 0) && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-gray-500 text-center">
                  <div className="text-lg mb-2">
                    {hasError ? 'Unable to connect to tracking system' : 'No objects detected'}
                  </div>
                  <div className="text-sm">
                    {hasError ? 'Check API connection' : 'Objects will appear here when detected by the CV system'}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Coordinate indicators */}
          <div className="absolute -bottom-6 left-0 text-xs text-gray-500">
            (0, 0)
          </div>
          <div className="absolute -bottom-6 right-0 text-xs text-gray-500">
            ({warehouseConfig.width_meters}, 0)
          </div>
          <div className="absolute -top-4 left-0 text-xs text-gray-500">
            (0, {warehouseConfig.length_meters})
          </div>
          <div className="absolute -top-4 right-0 text-xs text-gray-500">
            ({warehouseConfig.width_meters}, {warehouseConfig.length_meters})
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 flex justify-center">
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Object Status</h3>
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
              <span>New (&lt;5s)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
              <span>Tracking (5s-1m)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-teal-400"></div>
              <span>Established (&gt;1m)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WarehouseView;
