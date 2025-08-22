import React, { useState, useEffect } from 'react';
import { Camera, Maximize2, Play, Square, AlertCircle } from 'lucide-react';

interface CameraFeedProps {
  cameraId: number;
  onExpand: (cameraId: number) => void;
  palletCount: number;
  autoInit?: boolean; // when false, do not auto-connect or fetch until enabled
}

const CameraFeed: React.FC<CameraFeedProps> = ({ cameraId, onExpand, palletCount, autoInit = true }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  const cameraColors: Record<number, string> = {
    1: 'border-purple-600',
    2: 'border-blue-600',
    3: 'border-green-600',
    4: 'border-amber-600'
  };

  // Initialize only when autoInit is enabled
  useEffect(() => {
    if (!autoInit) return;
    checkCameraStatus();
    // Pull a single snapshot as lightweight thumbnail, then pause
    setThumbnailUrl(`http://localhost:8000/api/cameras/${cameraId}/snapshot?ts=${Date.now()}`);
  }, [cameraId, autoInit]);

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
      setPlaying(true);
      setIsConnected(true);
      setError(null);
    } catch (err) {
      setError('Connection failed');
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectCamera = async () => {
    try {
      setPlaying(false);
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
        {playing ? (
          <img
            src={`http://localhost:8000/api/cameras/${cameraId}/stream`}
            alt={`Camera ${cameraId} feed`}
            className="w-full h-full object-cover"
            onError={() => setError('Stream error')}
          />
        ) : (
          thumbnailUrl ? (
            <img
              src={thumbnailUrl}
              alt={`Camera ${cameraId} snapshot`}
              className="w-full h-full object-cover opacity-90"
              onError={() => setError('Snapshot error')}
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
              </div>
            </div>
          )
        )}

        {/* Play overlay when paused */}
        {!playing && (
          <button
            onClick={connectCamera}
            className="absolute bottom-2 right-2 bg-blue-600/90 hover:bg-blue-600 text-white text-xs px-2 py-1 rounded shadow"
            disabled={isConnecting}
          >
            {isConnecting ? 'Starting…' : 'Play'}
          </button>
        )}

        {/* Stop button when playing */}
        {playing && (
          <button
            onClick={disconnectCamera}
            className="absolute bottom-2 right-2 bg-red-600/90 hover:bg-red-600 text-white text-xs px-2 py-1 rounded shadow"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
};

export default CameraFeed;