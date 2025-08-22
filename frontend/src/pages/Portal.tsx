import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../index.css';
import WorkingWarehouseView from '../components/WorkingWarehouseView';

// Icons (using text for simplicity, same as backup)
const Package = () => <span>📦</span>;
const Activity = () => <span>📊</span>;
const Clock = () => <span>⏰</span>;
const TrendingUp = () => <span>📈</span>;
const Camera = () => <span>📹</span>;
const Search = () => <span>🔍</span>;
const Bell = () => <span>🔔</span>;
const User = () => <span>👤</span>;

// Stat Card Component
const StatCard = ({ icon, label, value, trend, trendLabel, color }: any) => {
  const colorClasses: any = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
    red: 'text-red-400'
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <div className={`${colorClasses[color]} text-lg`}>{icon}</div>
        {trend && (
          <div className="text-xs text-green-400">
            +{trend}% {trendLabel}
          </div>
        )}
      </div>
      <div className={`text-2xl font-bold ${colorClasses[color]} mb-1`}>{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  );
};

// Object Details Sidebar Component
const ObjectDetailsSidebar = ({ object, isOpen, onClose }: any) => {
  if (!isOpen || !object) return null;

  const formatTimestamp = (timestamp: any) => {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return 'Invalid date';
    }
  };

  return (
    <div className={`fixed right-0 top-0 h-full w-80 bg-gray-800 border-l border-gray-700 transform transition-transform duration-300 ease-in-out z-50 ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}>
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Pallet Details</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">✕</button>
      </div>
      <div className="p-4 overflow-y-auto h-full pb-20">
        <div className="space-y-4">
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Object ID</div>
            <div className="text-xl font-bold text-blue-400">{object.persistent_id || object.global_id || 'Unknown'}</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">🏷️ Warp ID</div>
            <div className="flex items-center gap-2">
              {object.warp_id ? (
                <>
                  <span className="text-green-400 font-mono font-semibold bg-green-900/20 px-2 py-1 rounded text-sm">{object.warp_id}</span>
                  <div className="w-2 h-2 bg-green-400 rounded-full" title="QR Code Linked"></div>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" title="Awaiting Robot Assignment"></div>
                  <span className="text-orange-400 font-medium">Not assigned yet</span>
                </>
              )}
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Camera</div>
            <div className="text-white font-medium">Zone {object.camera_id || 'Unknown'}</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="space-y-3">
              <div>
                <div className="text-sm text-gray-400 mb-1">Inbound Time</div>
                <div className="text-white text-sm">{formatTimestamp(object.first_seen || object.timestamp)}</div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Last Seen</div>
                <div className="text-white text-sm">{formatTimestamp(object.last_seen || object.timestamp)}</div>
              </div>
              {object.warp_id_linked_at && (
                <div>
                  <div className="text-sm text-gray-400 mb-1">Warp Linked</div>
                  <div className="text-purple-400 text-sm">{formatTimestamp(object.warp_id_linked_at)}</div>
                </div>
              )}
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">🎯 Detection Quality</div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-300">Confidence:</span>
                <span className="text-green-400 font-medium">{object.confidence ? `${(object.confidence * 100).toFixed(1)}%` : 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Area:</span>
                <span className="text-white">{object.area?.toLocaleString() || 'N/A'} px</span>
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">📍 Coordinates</div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-300">Physical:</span>
                <span className="text-cyan-400 font-medium">({object.physical_x_ft?.toFixed(1) || 'N/A'}, {object.physical_y_ft?.toFixed(1) || 'N/A'}) ft</span>
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-2">🔄 Tracking</div>
            <div className="space-y-2">
              <div className="flex justify-between"><span className="text-gray-300">Times Seen:</span><span className="text-white">{object.times_seen || 1}</span></div>
              {object.similarity_score && (
                <div className="flex justify-between">
                  <span className="text-gray-300">Similarity:</span>
                  <span className="text-blue-400 font-medium">{(object.similarity_score * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>
          </div>
          <details className="bg-gray-700 rounded-lg">
            <summary className="p-3 cursor-pointer text-sm text-gray-400 hover:text-white">🔧 Raw Data</summary>
            <div className="px-3 pb-3"><pre className="text-xs text-gray-300 bg-gray-800 p-2 rounded overflow-x-auto">{JSON.stringify(object, null, 2)}</pre></div>
          </details>
        </div>
      </div>
    </div>
  );
};

const getObjectColor = (obj: any) => {
  if (obj.color_rgb && Array.isArray(obj.color_rgb) && obj.color_rgb.length >= 3) {
    return `rgb(${obj.color_rgb[0]}, ${obj.color_rgb[1]}, ${obj.color_rgb[2]})`;
  }
  if (obj.color_hex) return obj.color_hex;
  if (obj.color_name && obj.color_confidence && obj.color_confidence > 0.3) {
    const colorMap: any = { red: '#ff4444', orange: '#ff8800', yellow: '#ffdd00', green: '#44ff44', blue: '#4444ff', purple: '#8844ff', pink: '#ff44aa', brown: '#8b4513', black: '#333333', white: '#f0f0f0', gray: '#888888', grey: '#888888', dark: '#444444' };
    const detected = colorMap[(obj.color_name as string).toLowerCase()];
    if (detected) return detected;
  }
  return '#d97706';
};

// Portal Component (original LiveWarehouse)
const Portal: React.FC = () => {
  const navigate = useNavigate();
  const [objects, setObjects] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [cameras, setCameras] = useState<any[]>([]);
  const [warehouseConfig, setWarehouseConfig] = useState({ width_feet: 180.0, length_feet: 90.0, width_meters: 54.864, length_meters: 27.432 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedObject, setSelectedObject] = useState<any>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightedObject, setHighlightedObject] = useState<any>(null);

  const handleObjectClick = (obj: any) => { setSelectedObject(obj); setSidebarOpen(true); };
  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (!query.trim()) { setHighlightedObject(null); return; }
    const found = objects.find(obj => obj.persistent_id?.toString().includes(query.trim()) || obj.global_id?.toString().includes(query.trim()));
    if (found) { setHighlightedObject(found); setSelectedObject(found); setSidebarOpen(true); } else { setHighlightedObject(null); }
  };
  const clearSearch = () => { setSearchQuery(''); setHighlightedObject(null); };
  const handleBackgroundClick = () => { setSelectedObject(null); setSidebarOpen(false); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && sidebarOpen) { setSelectedObject(null); setSidebarOpen(false); } };
    document.addEventListener('keydown', onKey); return () => document.removeEventListener('keydown', onKey);
  }, [sidebarOpen]);

  const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const fetchObjects = async () => {
    if (isPolling) return; setIsPolling(true);
    try {
      const resp = await fetch(`${API}/api/tracking/objects`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json(); setObjects(data.objects || []); setError(null);
    } catch (err) {
      console.error('❌ Fetch error:', err); setError('Connection failed'); setObjects([]);
    } finally { setLoading(false); setIsPolling(false); }
  };

  const fetchStats = async () => { try { const r = await fetch(`${API}/api/tracking/stats`); if (r.ok) setStats(await r.json()); } catch (e) { console.error('Stats fetch failed:', e); } };
  const fetchCameras = async () => { try { const r = await fetch(`${API}/api/cameras/status`); if (r.ok) { const d = await r.json(); setCameras(d.cameras || []); } } catch (e) { console.error('Camera status fetch failed:', e); } };
  const fetchWarehouseConfig = async () => { try { const r = await fetch(`${API}/api/warehouse/config`); if (r.ok) { const d = await r.json(); setWarehouseConfig({ width_feet: d.width_feet || 180, length_feet: d.length_feet || 90, width_meters: d.width_meters || 54.864, length_meters: d.length_meters || 27.432 }); } } catch (e) { console.error('Warehouse config fetch failed:', e); } };

  useEffect(() => {
    fetchObjects(); fetchStats(); fetchCameras(); fetchWarehouseConfig();
    const i1 = setInterval(() => { fetchObjects(); fetchStats(); }, 2000);
    const i2 = setInterval(fetchCameras, 10000);
    const i3 = setInterval(fetchWarehouseConfig, 30000);
    return () => { clearInterval(i1); clearInterval(i2); clearInterval(i3); };
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <img src="/logo3.png" alt="Logo" className="h-12 w-auto mr-4" onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
          </div>
          <div className="flex-1 flex flex-col items-center">
            <h2 className="text-2xl md:text-2xl font-extrabold text-center tracking-tight bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
              Digital Twin
              <span className="align-middle ml-3 text-sm md:text-base font-semibold text-gray-300 bg-clip-text text-transparent bg-gradient-to-r from-gray-300 to-gray-400">— Live Warehouse Tracking</span>
            </h2>
            <div className="flex gap-4 text-sm text-gray-400 mt-1">
              <span className="flex items-center gap-2"><div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>Live CV System Connected</span>
              <span>Objects: {objects.length}{selectedObject ? ` | Selected: ${selectedObject.persistent_id || selectedObject.global_id}` : ''}</span>
              <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-gray-500 rounded-full"></div>Warehouse: 180ft × 90ft</span>
              {error && <span className="text-red-400">Error: {error}</span>}
            </div>
          </div>
          {/* Right Side Buttons */}
          <div className="flex items-center space-x-2">
            <button className="p-2 hover:bg-gray-700 rounded-lg" title="Search"><Search /></button>
            <button className="p-2 hover:bg-gray-700 rounded-lg" title="Notifications"><Bell /></button>
            <button className="p-2 hover:bg-gray-700 rounded-lg" title="User"><User /></button>
            <button className="px-3 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white text-sm" onClick={() => navigate('/config')}>Configuration</button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-6">
        <WorkingWarehouseView />
      </main>
    </div>
  );
};

export default Portal;

