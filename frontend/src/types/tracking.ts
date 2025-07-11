export interface TrackedObject {
  persistent_id: number;
  warp_id?: string | null; // NEW: Warp ID from QR code
  center: {
    x: number;
    y: number;
  };
  real_center?: [number, number] | null;
  bbox: [number, number, number, number]; // [xmin, ymin, xmax, ymax]
  confidence: number;
  age_seconds: number;
  times_seen: number;
  status?: 'new' | 'tracking' | 'established';
  first_seen: string;
  last_seen: string;
  created_at?: string;
  updated_at?: string;
  warp_id_linked_at?: string; // NEW: When Warp ID was linked
}

export interface WarehouseConfig {
  // Primary dimensions in feet
  width_feet: number;
  length_feet: number;
  
  // Secondary dimensions in meters (for backward compatibility)
  width_meters: number;
  length_meters: number;
  
  calibrated: boolean;
  last_updated: string;
  units?: string;
}

export interface TrackingStats {
  total_detections: number;
  unique_objects: number;
  recent_objects: number;
  database_connected: boolean;
  timestamp: string;
}

export interface TrackingResponse {
  objects: TrackedObject[];
  count: number;
  timestamp: string;
}

export interface ObjectPosition {
  x: number; // percentage (0-100)
  y: number; // percentage (0-100)
}

// NEW: Warp ID related interfaces
export interface WarpIdLinkRequest {
  warp_id: string;
}

export interface WarpIdResponse {
  success: boolean;
  message: string;
  persistent_id?: number;
  global_id?: number;
  warp_id?: string;
}

export interface WarpIdSearchResponse {
  success: boolean;
  object?: TrackedObject;
  timestamp: string;
}

export interface WarpIdListResponse {
  success: boolean;
  count: number;
  objects: TrackedObject[];
  timestamp: string;
}
