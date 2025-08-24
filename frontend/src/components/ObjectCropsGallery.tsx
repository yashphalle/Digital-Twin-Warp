import React, { useEffect, useState } from 'react';

interface CropItem {
  url: string;
  mtime_ms: number;
  camera_id: number;
  date: string;
  filename: string;
}

interface Props {
  persistentId: number;
  limit?: number;
}

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const ObjectCropsGallery: React.FC<Props> = ({ persistentId, limit = 5 }) => {
  const [items, setItems] = useState<CropItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCrops = async () => {
    if (!persistentId) return;
    try {
      const resp = await fetch(`${API}/api/crops/${persistentId}?limit=${limit}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setItems((data.items || []) as CropItem[]);
      setError(null);
    } catch (e) {
      setError('Failed to load crops');
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { setLoading(true); fetchCrops(); const t = setInterval(fetchCrops, 10000); return () => clearInterval(t); }, [persistentId]);

  if (loading) return <div className="text-gray-400 text-sm">Loading...</div>;
  if (error) return <div className="text-red-400 text-sm">{error}</div>;
  if (!items.length) return <div className="text-gray-400 text-sm">No crops yet</div>;

  return (
    <div className="grid grid-cols-3 gap-2">
      {items.map((it, idx) => (
        <div key={`${it.camera_id}-${it.mtime_ms}-${idx}`} className="bg-gray-800 rounded overflow-hidden border border-gray-700">
          <img src={it.url.startsWith('http') ? it.url : `${API}${it.url}`} alt={it.filename} className="w-full h-20 object-cover" />
          <div className="p-1 text-[10px] text-gray-400 flex justify-between">
            <span>Cam {it.camera_id}</span>
            <span>{new Date(it.mtime_ms).toLocaleTimeString()}</span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ObjectCropsGallery;

