import React, { useState, useEffect } from 'react';
import { Camera, Maximize2, Play, Square, AlertCircle } from 'lucide-react';

interface CameraFeedProps {
  cameraId: number;
  onExpand: (cameraId: number) => void;
  palletCount: number;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ cameraId, onExpand, palletCount }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cameraColors: Record<number, string> = {
    1: 'border-purple-600',
    2: 'border-blue-600',
    3: 'border-green-600',
    4: 'border-amber-600'
  };

  // Check camera status on component mount
  useEffect(() => {
    checkCameraStatus();
  }, [cameraId]);

  const checkCameraStatus = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/camera/${cameraId}/status`);
      const data = await response.json();
      setIsConnected(data.status === 'connected');
      setError(data.status === 'error' ? 'Camera error' : null);
    } catch (err) {
      setError('Failed to check camera status');
    }
  };

  const connectCamera = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/camera/${cameraId}/connect`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsConnected(true);
        setError(null);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to connect');
      }
    } catch (err) {
      setError('Connection failed');
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectCamera = async () => {
    try {
      await fetch(`http://localhost:8000/api/camera/${cameraId}/disconnect`, {
        method: 'POST',
      });
      setIsConnected(false);
      setError(null);
    } catch (err) {
      setError('Failed to disconnect');
    }
  };

  return (
    <div className={`bg-gray-800 rounded-lg border-2 ${cameraColors[cameraId]} overflow-hidden`}>
      <div className="p-2 bg-gray-900 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium">Camera {cameraId}</span>
          <span className="text-xs text-gray-500">({palletCount} pallets)</span>
          {isConnected && <div className="w-2 h-2 bg-green-500 rounded-full"></div>}
          {error && <AlertCircle className="w-4 h-4 text-red-500" />}
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={isConnected ? disconnectCamera : connectCamera}
            disabled={isConnecting}
            className="p-1 hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
            title={isConnected ? 'Disconnect' : 'Connect'}
          >
            {isConnecting ? (
              <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
            ) : isConnected ? (
              <Square className="w-4 h-4 text-red-400" />
            ) : (
              <Play className="w-4 h-4 text-green-400" />
            )}
          </button>
          <button
            onClick={() => onExpand(cameraId)}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
          >
            <Maximize2 className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>
      <div className="p-0 h-48 bg-gray-900/50 relative">
        {isConnected ? (
          <img
            src={`http://localhost:8000/api/camera/${cameraId}/stream`}
            alt={`Camera ${cameraId} feed`}
            className="w-full h-full object-cover"
            onError={() => setError('Stream error')}
          />
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              {error ? (
                <>
                  <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
                  <p className="text-sm text-red-400">{error}</p>
                </>
              ) : (
                <>
                  <Camera className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                  <p className="text-sm text-gray-400">Camera disconnected</p>
                </>
              )}
              <p className="text-xs text-gray-500 mt-1">Zone {cameraId}</p>
              {!isConnecting && (
                <button
                  onClick={connectCamera}
                  className="mt-3 px-3 py-1 bg-blue-600 text-xs rounded hover:bg-blue-700 transition-colors"
                >
                  Connect
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraFeed;