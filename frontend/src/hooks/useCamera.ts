import { useState, useEffect, useRef, useCallback } from 'react';

export interface CameraDevice {
  deviceId: string;
  label: string;
  kind: string;
}

export interface CameraError {
  code: string;
  message: string;
}

export interface UseCameraOptions {
  deviceId?: string;
  width?: number;
  height?: number;
  frameRate?: number;
}

export interface UseCameraReturn {
  stream: MediaStream | null;
  isLoading: boolean;
  error: CameraError | null;
  devices: CameraDevice[];
  startCamera: (options?: UseCameraOptions) => Promise<void>;
  stopCamera: () => void;
  switchDevice: (deviceId: string) => Promise<void>;
  isSupported: boolean;
}

export const useCamera = (): UseCameraReturn => {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<CameraError | null>(null);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Check if camera API is supported
  const isSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  // Get available camera devices
  const getDevices = useCallback(async () => {
    if (!isSupported) return;

    try {
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = deviceList
        .filter(device => device.kind === 'videoinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
          kind: device.kind
        }));
      
      setDevices(videoDevices);
    } catch (err) {
      console.error('Failed to enumerate devices:', err);
      setError({
        code: 'DEVICE_ENUMERATION_FAILED',
        message: 'Failed to get camera devices'
      });
    }
  }, [isSupported]);

  // Start camera with given options
  const startCamera = useCallback(async (options: UseCameraOptions = {}) => {
    if (!isSupported) {
      setError({
        code: 'NOT_SUPPORTED',
        message: 'Camera API is not supported in this browser'
      });
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: options.width || 640 },
          height: { ideal: options.height || 480 },
          frameRate: { ideal: options.frameRate || 30 },
          ...(options.deviceId && { deviceId: { exact: options.deviceId } })
        },
        audio: false
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = newStream;
      setStream(newStream);
      setIsLoading(false);

      // Refresh device list after getting permission
      await getDevices();
    } catch (err) {
      setIsLoading(false);
      
      let errorCode = 'UNKNOWN_ERROR';
      let errorMessage = 'Failed to start camera';

      if (err instanceof Error) {
        switch (err.name) {
          case 'NotAllowedError':
            errorCode = 'PERMISSION_DENIED';
            errorMessage = 'Camera permission denied. Please allow camera access and try again.';
            break;
          case 'NotFoundError':
            errorCode = 'NO_CAMERA_FOUND';
            errorMessage = 'No camera device found. Please connect a camera and try again.';
            break;
          case 'NotReadableError':
            errorCode = 'CAMERA_IN_USE';
            errorMessage = 'Camera is already in use by another application.';
            break;
          case 'OverconstrainedError':
            errorCode = 'CONSTRAINTS_NOT_SATISFIED';
            errorMessage = 'Camera does not support the requested settings.';
            break;
          default:
            errorMessage = err.message;
        }
      }

      setError({ code: errorCode, message: errorMessage });
    }
  }, [isSupported, getDevices]);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setStream(null);
    setError(null);
  }, []);

  // Switch to a different camera device
  const switchDevice = useCallback(async (deviceId: string) => {
    await startCamera({ deviceId });
  }, [startCamera]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Get devices on mount
  useEffect(() => {
    getDevices();
  }, [getDevices]);

  return {
    stream,
    isLoading,
    error,
    devices,
    startCamera,
    stopCamera,
    switchDevice,
    isSupported
  };
};
