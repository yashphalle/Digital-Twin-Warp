import { useState } from 'react';
import { WarpIdLinkRequest, WarpIdResponse, WarpIdSearchResponse, WarpIdListResponse } from '../types/tracking';

const API_BASE = (import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000');
const API_BASE_URL = `${API_BASE}/api`;

export const useWarpId = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const linkWarpId = async (persistentId: number, warpId: string): Promise<WarpIdResponse> => {
    setLoading(true);
    setError(null);
    
    try {
      const request: WarpIdLinkRequest = { warp_id: warpId };
      
      const response = await fetch(`${API_BASE_URL}/tracking/objects/${persistentId}/warp-id`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: WarpIdResponse = await response.json();
      
      if (!data.success) {
        setError(data.message);
      }
      
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to link Warp ID';
      setError(errorMessage);
      return {
        success: false,
        message: errorMessage
      };
    } finally {
      setLoading(false);
    }
  };

  const searchByWarpId = async (warpId: string): Promise<WarpIdSearchResponse | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/tracking/objects/by-warp-id/${encodeURIComponent(warpId)}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          return {
            success: false,
            timestamp: new Date().toISOString()
          };
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: WarpIdSearchResponse = await response.json();
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to search by Warp ID';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const getAllWarpIds = async (): Promise<WarpIdListResponse | null> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/tracking/warp-ids`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: WarpIdListResponse = await response.json();
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch Warp IDs';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return {
    linkWarpId,
    searchByWarpId,
    getAllWarpIds,
    loading,
    error,
    clearError: () => setError(null)
  };
};
