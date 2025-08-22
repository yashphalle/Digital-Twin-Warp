import { useState, useEffect } from 'react';

// Define types inline to avoid import issues
interface TrackedObject {
  persistent_id: number;
  warp_id?: string | null;
  center: {
    x: number;
    y: number;
  };
  real_center?: [number, number] | null;
  bbox: [number, number, number, number];
  confidence: number;
  age_seconds: number;
  times_seen: number;
  status?: 'new' | 'tracking' | 'established';
  first_seen: string;
  last_seen: string;
  created_at?: string;
  updated_at?: string;
  warp_id_linked_at?: string;
}

interface WarehouseConfig {
  width_feet: number;
  length_feet: number;
  width_meters: number;
  length_meters: number;
  calibrated: boolean;
  last_updated: string;
  units?: string;
}

interface TrackingStats {
  total_detections: number;
  unique_objects: number;
  recent_objects: number;
  database_connected: boolean;
  timestamp: string;
}

interface TrackingResponse {
  objects: TrackedObject[];
  count: number;
  timestamp: string;
}

const API_BASE_URL = 'http://localhost:8000/api';

export const useTracking = () => {
  const [objects, setObjects] = useState<TrackedObject[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchObjects = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tracking/objects`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TrackingResponse = await response.json();
      setObjects(data.objects || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch objects');
      console.error('Error fetching tracking objects:', err);
      // Set empty array on error so component can still render
      setObjects([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchObjects();
    // Refresh every 5 seconds for now (later we'll use WebSocket)
    const interval = setInterval(fetchObjects, 5000);
    return () => clearInterval(interval);
  }, []);

  return {
    objects,
    loading,
    error,
    refetch: fetchObjects
  };
};

export const useWarehouseConfig = () => {
  const [config, setConfig] = useState<WarehouseConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/warehouse/config`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: WarehouseConfig = await response.json();
        setConfig(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch warehouse config');
        console.error('Error fetching warehouse config:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  return {
    config,
    loading,
    error
  };
};

export const useTrackingStats = () => {
  const [stats, setStats] = useState<TrackingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tracking/stats`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TrackingStats = await response.json();
      setStats(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch tracking stats');
      console.error('Error fetching tracking stats:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    // Refresh every 5 seconds
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return {
    stats,
    loading,
    error,
    refetch: fetchStats
  };
};
