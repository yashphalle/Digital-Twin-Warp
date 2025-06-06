import React, { useRef, useEffect, useState } from 'react';
import { Camera, Maximize2, Play, Square, Settings, AlertTriangle } from 'lucide-react';
import { useCamera } from '../hooks/useCamera';

interface CameraFeedProps {
  cameraId: number;
  onExpand: (cameraId: number) => void;
  palletCount: number;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ cameraId, onExpand, palletCount }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  const {
    stream,
    isLoading,
    error,
    devices,
    startCamera,
    stopCamera,
    switchDevice,
    isSupported
  } = useCamera();

  const cameraColors: Record<number, string> = {
    1: 'border-purple-600',
    2: 'border-blue-600',
    3: 'border-green-600',
    4: 'border-amber-600'
  };

  // Auto-select device based on camera ID
  useEffect(() => {
    if (devices.length > 0 && !selectedDeviceId) {
      // Try to assign different devices to different cameras
      const deviceIndex = Math.min(cameraId - 1, devices.length - 1);
      setSelectedDeviceId(devices[deviceIndex].deviceId);
    }
  }, [devices, cameraId, selectedDeviceId]);

  // Update video element when stream changes
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const handleToggleCamera = async () => {
    if (isActive) {
      stopCamera();
      setIsActive(false);
    } else {
      await startCamera({ deviceId: selectedDeviceId });
      setIsActive(true);
    }
  };

  const handleDeviceChange = async (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    if (isActive) {
      await switchDevice(deviceId);
    }
  };

  const getStatusIndicator = () => {
    if (!isSupported) {
      return <AlertTriangle className="w-2 h-2 text-red-500" />;
    }
    if (isLoading) {
      return <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />;
    }
    if (error) {
      return <div className="w-2 h-2 bg-red-500 rounded-full" />;
    }
    if (isActive && stream) {
      return <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />;
    }
    return <div className="w-2 h-2 bg-gray-500 rounded-full" />;
  };

  const renderVideoContent = () => {
    if (!isSupported) {
      return (
        <div className="h-full flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
            <p className="text-xs text-red-400">Camera not supported</p>
            <p className="text-xs text-red-500">Please use a modern browser</p>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="h-full flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
            <p className="text-xs text-red-400">Camera Error</p>
            <p className="text-xs text-red-500 max-w-32 break-words">{error.message}</p>
          </div>
        </div>
      );
    }

    if (isLoading) {
      return (
        <div className="h-full flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
            <p className="text-xs text-gray-400">Starting camera...</p>
          </div>
        </div>
      );
    }

    if (isActive && stream) {
      return (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }} // Mirror for natural feel
        />
      );
    }

    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Camera className="w-8 h-8 text-gray-600 mx-auto mb-2" />
          <p className="text-xs text-gray-500">Camera Off</p>
          <p className="text-xs text-gray-600">Zone {cameraId}</p>
        </div>
      </div>
    );
  };

  return (
    <div className={`bg-gray-800 rounded-lg border-2 ${cameraColors[cameraId]} overflow-hidden`}>
      {/* Header */}
      <div className="p-2 bg-gray-900 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium">Camera {cameraId}</span>
          <span className="text-xs text-gray-500">({palletCount} pallets)</span>
          {getStatusIndicator()}
        </div>
        <div className="flex items-center space-x-1">
          {isSupported && (
            <>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-1 hover:bg-gray-700 rounded transition-colors"
                title="Camera Settings"
              >
                <Settings className="w-3 h-3 text-gray-400" />
              </button>
              <button
                onClick={handleToggleCamera}
                className={`px-2 py-1 text-xs rounded transition-colors flex items-center space-x-1 ${
                  isActive 
                    ? 'bg-red-600 hover:bg-red-700 text-white' 
                    : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
                disabled={isLoading}
              >
                {isActive ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                <span>{isActive ? 'Stop' : 'Start'}</span>
              </button>
            </>
          )}
          <button
            onClick={() => onExpand(cameraId)}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
            title="Expand Camera"
          >
            <Maximize2 className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Video Content */}
      <div className="h-32 bg-gray-900/50 relative">
        {renderVideoContent()}
      </div>

      {/* Settings Panel */}
      {showSettings && devices.length > 0 && (
        <div className="p-2 bg-gray-900 border-t border-gray-700">
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Camera Device:</label>
            <select
              value={selectedDeviceId}
              onChange={(e) => handleDeviceChange(e.target.value)}
              className="w-full text-xs bg-gray-800 text-white border border-gray-600 rounded px-2 py-1"
              disabled={isLoading}
            >
              {devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label}
                </option>
              ))}
            </select>
            {devices.length > 1 && (
              <p className="text-xs text-gray-500">
                {devices.length} camera{devices.length > 1 ? 's' : ''} available
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraFeed;
