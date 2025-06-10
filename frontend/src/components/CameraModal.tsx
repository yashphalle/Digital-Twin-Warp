import React from 'react';
import { X, Camera } from 'lucide-react';

interface CameraModalProps {
  cameraId: number;
  onClose: () => void;
}

const CameraModal: React.FC<CameraModalProps> = ({ cameraId, onClose }) => {
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
        
        <div className="p-0 h-[600px] flex items-center justify-center bg-gray-900/50">
          <div className="text-center">
            <p className="text-lg text-gray-400">Camera feed not connected</p>
            <p className="text-sm text-gray-500 mt-2">Zone {cameraId}</p>
            <button className="mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors">
              Connect
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraModal;