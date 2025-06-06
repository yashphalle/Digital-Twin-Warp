import React, { useRef, useEffect } from 'react';
import { X, Camera, Settings, Download, Maximize2, Minimize2 } from 'lucide-react';
import { useCamera } from '../hooks/useCamera';

interface CameraModalProps {
  cameraId: number;
  isOpen: boolean;
  onClose: () => void;
}

const CameraModal: React.FC<CameraModalProps> = ({ cameraId, isOpen, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isFullscreen, setIsFullscreen] = React.useState(false);
  const [selectedDeviceId, setSelectedDeviceId] = React.useState<string>('');

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
    1: 'border-purple-600 bg-purple-900/10',
    2: 'border-blue-600 bg-blue-900/10',
    3: 'border-green-600 bg-green-900/10',
    4: 'border-amber-600 bg-amber-900/10'
  };

  // Auto-select device and start camera when modal opens
  useEffect(() => {
    if (isOpen && devices.length > 0 && !selectedDeviceId) {
      const deviceIndex = Math.min(cameraId - 1, devices.length - 1);
      const deviceId = devices[deviceIndex].deviceId;
      setSelectedDeviceId(deviceId);
      startCamera({ deviceId, width: 1280, height: 720 });
    }
  }, [isOpen, devices, cameraId, selectedDeviceId, startCamera]);

  // Update video element when stream changes
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // Stop camera when modal closes
  useEffect(() => {
    if (!isOpen) {
      stopCamera();
    }
  }, [isOpen, stopCamera]);

  // Handle device change
  const handleDeviceChange = async (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    await switchDevice(deviceId);
  };

  // Handle fullscreen toggle
  const handleFullscreenToggle = () => {
    if (!isFullscreen && videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
        setIsFullscreen(true);
      }
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Handle screenshot
  const handleScreenshot = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      
      if (ctx) {
        // Flip horizontally to match the mirrored display
        ctx.scale(-1, 1);
        ctx.drawImage(videoRef.current, -canvas.width, 0);
        
        // Download the image
        canvas.toBlob((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `camera-${cameraId}-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
          }
        });
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className={`bg-gray-900 rounded-lg border-2 ${cameraColors[cameraId]} max-w-4xl max-h-[90vh] w-full mx-4 flex flex-col`}>
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Camera className="w-6 h-6 text-gray-400" />
            <h2 className="text-xl font-semibold text-white">Camera {cameraId} - Expanded View</h2>
            {stream && (
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-green-400">Live</span>
              </div>
            )}
          </div>
          <div className="flex items-center space-x-2">
            {stream && (
              <>
                <button
                  onClick={handleScreenshot}
                  className="p-2 hover:bg-gray-700 rounded transition-colors"
                  title="Take Screenshot"
                >
                  <Download className="w-5 h-5 text-gray-400" />
                </button>
                <button
                  onClick={handleFullscreenToggle}
                  className="p-2 hover:bg-gray-700 rounded transition-colors"
                  title="Toggle Fullscreen"
                >
                  {isFullscreen ? (
                    <Minimize2 className="w-5 h-5 text-gray-400" />
                  ) : (
                    <Maximize2 className="w-5 h-5 text-gray-400" />
                  )}
                </button>
              </>
            )}
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-700 rounded transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>
        </div>

        {/* Video Content */}
        <div className="flex-1 p-4">
          <div className="relative bg-gray-800 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
            {!isSupported ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <p className="text-lg text-red-400">Camera not supported</p>
                  <p className="text-sm text-red-500">Please use a modern browser</p>
                </div>
              </div>
            ) : error ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <p className="text-lg text-red-400">Camera Error</p>
                  <p className="text-sm text-red-500 max-w-md">{error.message}</p>
                </div>
              </div>
            ) : isLoading ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-lg text-gray-400">Starting camera...</p>
                </div>
              </div>
            ) : stream ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ transform: 'scaleX(-1)' }}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <p className="text-lg text-gray-500">Camera Off</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        {devices.length > 0 && (
          <div className="p-4 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Settings className="w-5 h-5 text-gray-400" />
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-400">Camera Device:</label>
                  <select
                    value={selectedDeviceId}
                    onChange={(e) => handleDeviceChange(e.target.value)}
                    className="bg-gray-800 text-white border border-gray-600 rounded px-3 py-1 text-sm"
                    disabled={isLoading}
                  >
                    {devices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="text-sm text-gray-500">
                {stream && videoRef.current && (
                  <span>
                    Resolution: {videoRef.current.videoWidth} × {videoRef.current.videoHeight}
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraModal;
