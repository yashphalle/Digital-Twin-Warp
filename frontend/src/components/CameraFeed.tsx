import React from 'react';
import { Camera, Maximize2 } from 'lucide-react';

interface CameraFeedProps {
  cameraId: number;
  onExpand: (cameraId: number) => void;
  palletCount: number;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ cameraId, onExpand, palletCount }) => {
  const cameraColors: Record<number, string> = {
    1: 'border-purple-600',
    2: 'border-blue-600',
    3: 'border-green-600',
    4: 'border-amber-600'
  };
  
  return (
    <div className={`bg-gray-800 rounded-lg border-2 ${cameraColors[cameraId]} overflow-hidden`}>
      <div className="p-2 bg-gray-900 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium">Camera {cameraId}</span>
          <span className="text-xs text-gray-500">({palletCount} pallets)</span>
        </div>
        <button
          onClick={() => onExpand(cameraId)}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
        >
          <Maximize2 className="w-4 h-4 text-gray-400" />
        </button>
      </div>
      <div className="p-0 h-48 flex items-center justify-center bg-gray-900/50">
        <div className="text-center">
          <p className="text-sm text-gray-400">Camera feed not connected</p>
          <p className="text-xs text-gray-500 mt-1">Zone {cameraId}</p>
          <button className="mt-3 px-3 py-1 bg-blue-600 text-xs rounded hover:bg-blue-700 transition-colors">
            Connect
          </button>
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;