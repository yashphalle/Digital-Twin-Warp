import React, { useState, useEffect } from 'react';
import { X, Camera, AlertCircle } from 'lucide-react';

interface CameraModalProps {
  cameraId: number;
  isOpen: boolean;
  onClose: () => void;
}

const CameraModal: React.FC<CameraModalProps> = ({ cameraId, isOpen, onClose }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      checkCameraStatus();
    }
  }, [cameraId, isOpen]);

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
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg w-full max-w-4xl overflow-hidden">
        <div className="p-4 bg-gray-900 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Camera className="w-5 h-5 text-gray-400" />
            <h3 className="text-lg font-medium">Camera {cameraId} Feed</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
          >
            <X className="w-6 h-6 text-gray-400" />
          </button>
        </div>
        
        <div className="p-0 h-[600px] bg-gray-900/50 relative">
          {isConnected ? (
            <img
              src={`http://localhost:8000/api/camera/${cameraId}/stream`}
              alt={`Camera ${cameraId} feed`}
              className="w-full h-full object-contain"
              onError={() => setError('Stream error')}
            />
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                {error ? (
                  <>
                    <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                    <p className="text-lg text-red-400">{error}</p>
                  </>
                ) : (
                  <>
                    <Camera className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                    <p className="text-lg text-gray-400">Camera disconnected</p>
                  </>
                )}
                <p className="text-sm text-gray-500 mt-2">Zone {cameraId}</p>
                <p className="text-xs text-gray-600 mt-1">Connect camera from the main dashboard</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraModal;