import React, { useState } from 'react';
import { Settings, Save, RefreshCw, AlertTriangle, Database, Camera, Cpu, Lock, Grid3x3, UploadCloud, ArrowLeft } from 'lucide-react';

interface ConfigurationTabProps {
  isAuthenticated: boolean;
}

const ConfigurationTab: React.FC<ConfigurationTabProps> = ({ isAuthenticated }) => {
  const [activeSection, setActiveSection] = useState<null | 'warehouse' | 'cv' | 'database' | 'collection'>(null);
  const [isSaving, setIsSaving] = useState(false);

  // Sample configuration data - these would come from API calls
  const [warehouseConfig, setWarehouseConfig] = useState({
    width_feet: 180.0,
    length_feet: 90.0,
    width_meters: 54.864,
    length_meters: 27.432,
    camera_count: 11
  });

  const [cameraConfig, setCameraConfig] = useState({
    active_cameras: [1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
    frame_skip: 20,
    resize: [1080, 1920],
    use_local_cameras: false
  });

  const [detectionConfig, setDetectionConfig] = useState({
    model_path: "Custom-4.pt",
    confidence_threshold: 0.5,
    iou_threshold: 0.45,
    filter_area_min: 15000,
    filter_area_max: 1000000,
    reid_enabled: true,
    reid_similarity_threshold: 0.5
  });

  const [databaseConfig, setDatabaseConfig] = useState({
    db_enabled: true,
    db_uri: "mongodb+srv://yash:1234@cluster0.jmslb8o.mongodb.net/",
    db_database: "WARP",
    db_collection: "detections",
    redis_uri: "redis://localhost:6379/0"
  });
  const [cvConfig, setCvConfig] = useState({
    frame_sampling: 20,
    gpu_detection_workers: 2,
    batching_strategy: 'most_recent' as 'most_recent' | 'fifo',
    batch_size: 11,
    overwrite_when_full: true,
    precision: 'fp32' as 'fp32' | 'fp16',
    fisheye_correction: true,
    tracking: {
      iou_strong: 0.8,
      amb_topk: 1,
      persist_frames: 30,
      reuse_embeddings: true,
      similarity_threshold: 0.5,
    },
    sift: {
      enabled: true,
      async_threads: true,
      timeout_ms: 250,
    },
  });

  const [collectionConfig, setCollectionConfig] = useState({
    raw_4k: true,
    corrected_1080p: true,
    interval_minutes: 1,
    gui_grid: true,
    upload_roboflow: true,
    output_dir: '/data/collection'
  });


  const handleUnauthorizedAction = () => {
    alert('You need to login to modify configurations. Please use Admin Login to access this feature.');
  };

  const handleSave = async (section: string) => {
    if (!isAuthenticated) {
      handleUnauthorizedAction();
      return;
    }

    setIsSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsSaving(false);
    alert(`${section} configuration saved successfully!`);
  };

  const ConfigSection = ({ title, icon, children, sectionKey }: any) => (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-600/20 rounded-lg text-blue-400">
            {icon}
          </div>
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
        <button
          onClick={() => handleSave(sectionKey)}
          disabled={isSaving}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
            isAuthenticated
              ? 'bg-blue-600 hover:bg-blue-500 text-white'
              : 'bg-gray-600 text-gray-400 cursor-not-allowed'
          }`}
        >
          {isSaving ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Save className="w-4 h-4" />
          )}
          <span>Save</span>
        </button>
      </div>
      {children}
    </div>
  );

  const JsonEditor = ({ data, onChange, disabled = false }: any) => (
    <div className="relative">
      <textarea
        value={JSON.stringify(data, null, 2)}
        onChange={(e) => {
          if (disabled) {
            handleUnauthorizedAction();
            return;
          }
          try {
            const parsed = JSON.parse(e.target.value);
            onChange(parsed);
          } catch (err) {
            // Invalid JSON, don't update
          }
        }}
        className={`w-full h-48 bg-gray-900 border border-gray-600 rounded-lg p-4 text-sm font-mono text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none ${
          disabled ? 'cursor-not-allowed opacity-75' : ''
        }`}
        placeholder="Configuration JSON..."
      />
      {!isAuthenticated && (
        <div className="absolute inset-0 bg-gray-900/50 rounded-lg flex items-center justify-center">
          <div className="flex items-center space-x-2 text-yellow-400 bg-gray-800 px-4 py-2 rounded-lg border border-yellow-400/20">
            <Lock className="w-4 h-4" />
            <span className="text-sm font-medium">Login Required</span>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {activeSection !== null && (
            <button
              className="p-2 rounded-lg hover:bg-gray-700"
              onClick={() => setActiveSection(null)}
              title="Back"
            >
              <ArrowLeft className="w-5 h-5 text-gray-300" />
            </button>
          )}
          <Settings className="w-6 h-6 text-blue-400" />
          <h2 className="text-xl font-semibold">System Configuration</h2>
        </div>
        <div className="flex items-center space-x-2">
          {!isAuthenticated && (
            <div className="flex items-center space-x-2 text-yellow-400 bg-yellow-400/10 border border-yellow-400/20 rounded-lg px-3 py-2">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm">Read-only mode - Login to edit</span>
            </div>
          )}
        </div>
      </div>

      {/* Category Cards or Expanded Panels */}
      {activeSection === null ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <button onClick={() => setActiveSection('warehouse')} className="bg-gray-800 border border-gray-700 hover:border-blue-500/50 rounded-lg p-6 text-left transition-all">
            <div className="flex items-center space-x-3 mb-2"><Grid3x3 className="w-5 h-5 text-blue-400" /><h3 className="font-semibold">Warehouse Layout</h3></div>
            <p className="text-sm text-gray-400">Dimensions and on-screen overlays</p>
          </button>
          <button onClick={() => setActiveSection('cv')} className="bg-gray-800 border border-gray-700 hover:border-blue-500/50 rounded-lg p-6 text-left transition-all">
            <div className="flex items-center space-x-3 mb-2"><Cpu className="w-5 h-5 text-blue-400" /><h3 className="font-semibold">CV Pipeline</h3></div>
            <p className="text-sm text-gray-400">Sampling, batching, tracking, SIFT</p>
          </button>
          <button onClick={() => setActiveSection('database')} className="bg-gray-800 border border-gray-700 hover:border-blue-500/50 rounded-lg p-6 text-left transition-all">
            <div className="flex items-center space-x-3 mb-2"><Database className="w-5 h-5 text-blue-400" /><h3 className="font-semibold">Database</h3></div>
            <p className="text-sm text-gray-400">MongoDB and staleness monitor</p>
          </button>
          <button onClick={() => setActiveSection('collection')} className="bg-gray-800 border border-gray-700 hover:border-blue-500/50 rounded-lg p-6 text-left transition-all">
            <div className="flex items-center space-x-3 mb-2"><UploadCloud className="w-5 h-5 text-blue-400" /><h3 className="font-semibold">Data Collection</h3></div>
            <p className="text-sm text-gray-400">Raw + corrected captures and upload</p>
          </button>
        </div>
      ) : (
        <div>
          {activeSection === 'warehouse' && (
            <ConfigSection title="Warehouse Layout" icon={<Grid3x3 className="w-5 h-5" />} sectionKey="warehouse">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Width (meters)</label>
                  <input type="number" step="0.01" value={warehouseConfig.width_meters} onChange={(e)=>setWarehouseConfig({...warehouseConfig, width_meters: parseFloat(e.target.value)||0})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Length (meters)</label>
                  <input type="number" step="0.01" value={warehouseConfig.length_meters} onChange={(e)=>setWarehouseConfig({...warehouseConfig, length_meters: parseFloat(e.target.value)||0})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Show camera zones overlay</label>
                  <input type="checkbox" checked={true} onChange={()=>{}} className="mr-2" disabled />
                  <span className="text-xs text-gray-500">Always on (per your preference)</span>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Show origin markers</label>
                  <input type="checkbox" checked={false} onChange={()=>{}} className="mr-2" disabled />
                  <span className="text-xs text-gray-500">We’ve hidden these in the layout</span>
                </div>
              </div>
            </ConfigSection>
          )}

          {activeSection === 'cv' && (
            <ConfigSection title="CV Pipeline" icon={<Cpu className="w-5 h-5" />} sectionKey="cv">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Frame sampling (1 in N)</label>
                  <input type="number" min={1} value={cvConfig.frame_sampling} onChange={(e)=>setCvConfig({...cvConfig, frame_sampling: Math.max(1, parseInt(e.target.value||'1'))})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">GPU detection workers</label>
                  <input type="number" min={1} max={8} value={cvConfig.gpu_detection_workers} onChange={(e)=>setCvConfig({...cvConfig, gpu_detection_workers: Math.max(1, parseInt(e.target.value||'1'))})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Batching strategy</label>
                  <select value={cvConfig.batching_strategy} onChange={(e)=>setCvConfig({...cvConfig, batching_strategy: e.target.value as any})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated}>
                    <option value="most_recent">Always most recent (preferred)</option>
                    <option value="fifo">FIFO</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Batch size</label>
                  <input type="number" min={1} value={cvConfig.batch_size} onChange={(e)=>setCvConfig({...cvConfig, batch_size: Math.max(1, parseInt(e.target.value||'1'))})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm text-gray-400 mb-1">When queue is full</label>
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" checked={cvConfig.overwrite_when_full} onChange={(e)=>setCvConfig({...cvConfig, overwrite_when_full: e.target.checked})} disabled={!isAuthenticated} />
                    Replace old frames (preferred)
                  </label>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Precision</label>
                  <select value={cvConfig.precision} onChange={(e)=>setCvConfig({...cvConfig, precision: e.target.value as any})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated}>
                    <option value="fp32">FP32 (preferred)</option>
                    <option value="fp16">FP16</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Fisheye correction</label>
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" checked={cvConfig.fisheye_correction} onChange={(e)=>setCvConfig({...cvConfig, fisheye_correction: e.target.checked})} disabled={!isAuthenticated} />
                    Enable fisheye correction
                  </label>
                </div>
                <div className="md:col-span-2 border-t border-gray-700 pt-2">
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">Tracking (BoT-SORT)</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">IoU strong</label>
                      <input type="range" min={0.5} max={0.95} step={0.01} value={cvConfig.tracking.iou_strong} onChange={(e)=>setCvConfig({...cvConfig, tracking: {...cvConfig.tracking, iou_strong: parseFloat(e.target.value)}})} className="w-full" disabled={!isAuthenticated} />
                      <div className="text-xs text-gray-400 mt-1">{cvConfig.tracking.iou_strong.toFixed(2)}</div>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Association top-k</label>
                      <input type="number" min={1} max={5} value={cvConfig.tracking.amb_topk} onChange={(e)=>setCvConfig({...cvConfig, tracking: {...cvConfig.tracking, amb_topk: Math.max(1, parseInt(e.target.value||'1'))}})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Persist frames after disappear</label>
                      <input type="number" min={0} max={120} value={cvConfig.tracking.persist_frames} onChange={(e)=>setCvConfig({...cvConfig, tracking: {...cvConfig.tracking, persist_frames: Math.max(0, parseInt(e.target.value||'0'))}})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                    </div>
                    <div className="md:col-span-3">
                      <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                        <input type="checkbox" checked={cvConfig.tracking.reuse_embeddings} onChange={(e)=>setCvConfig({...cvConfig, tracking: {...cvConfig.tracking, reuse_embeddings: e.target.checked}})} disabled={!isAuthenticated} />
                        Reuse embeddings from detector
                      </label>
                    </div>
                    <div className="md:col-span-3">
                      <label className="block text-sm text-gray-400 mb-1">Appearance similarity threshold</label>
                      <input type="range" min={0} max={1} step={0.01} value={cvConfig.tracking.similarity_threshold} onChange={(e)=>setCvConfig({...cvConfig, tracking: {...cvConfig.tracking, similarity_threshold: parseFloat(e.target.value)}})} className="w-full" disabled={!isAuthenticated} />
                      <div className="text-xs text-gray-400 mt-1">{cvConfig.tracking.similarity_threshold.toFixed(2)}</div>
                    </div>
                  </div>
                </div>
                <div className="md:col-span-2 border-t border-gray-700 pt-2">
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">SIFT (advanced)</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                        <input type="checkbox" checked={cvConfig.sift.enabled} onChange={(e)=>setCvConfig({...cvConfig, sift: {...cvConfig.sift, enabled: e.target.checked}})} disabled={!isAuthenticated} />
                        Enable SIFT
                      </label>
                    </div>
                    <div>
                      <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                        <input type="checkbox" checked={cvConfig.sift.async_threads} onChange={(e)=>setCvConfig({...cvConfig, sift: {...cvConfig.sift, async_threads: e.target.checked}})} disabled={!isAuthenticated} />
                        Async threading
                      </label>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Timeout (ms)</label>
                      <input type="number" min={0} max={5000} value={cvConfig.sift.timeout_ms} onChange={(e)=>setCvConfig({...cvConfig, sift: {...cvConfig.sift, timeout_ms: Math.max(0, parseInt(e.target.value||'0'))}})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                    </div>
                  </div>
                </div>
              </div>
            </ConfigSection>
          )}

          {activeSection === 'database' && (
            <ConfigSection title="Database" icon={<Database className="w-5 h-5" />} sectionKey="database">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" checked={databaseConfig.db_enabled} onChange={(e)=>setDatabaseConfig({...databaseConfig, db_enabled: e.target.checked})} disabled={!isAuthenticated} />
                    MongoDB enabled
                  </label>
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm text-gray-400 mb-1">MongoDB URI</label>
                  <input type="password" value={databaseConfig.db_uri} onChange={(e)=>setDatabaseConfig({...databaseConfig, db_uri: e.target.value})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Database</label>
                  <input type="text" value={databaseConfig.db_database} onChange={(e)=>setDatabaseConfig({...databaseConfig, db_database: e.target.value})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Collection</label>
                  <input type="text" value={databaseConfig.db_collection} onChange={(e)=>setDatabaseConfig({...databaseConfig, db_collection: e.target.value})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Batch write interval (s)</label>
                  <input type="number" min={1} max={10} value={2} onChange={()=>{}} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Min consecutive detections before write</label>
                  <input type="number" min={1} max={10} value={5} onChange={()=>{}} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled />
                </div>
                <div className="md:col-span-2 border-t border-gray-700 pt-2">
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">Staleness monitor</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-1">Mark inactive after (minutes)</label>
                      <input type="number" min={1} max={60} value={1} onChange={()=>{}} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled />
                    </div>
                    <div className="md:col-span-2">
                      <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                        <input type="checkbox" checked={true} onChange={()=>{}} disabled />
                        Reactivate on next sighting; never remove records with warp ID
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </ConfigSection>
          )}

          {activeSection === 'collection' && (
            <ConfigSection title="Data Collection" icon={<UploadCloud className="w-5 h-5" />} sectionKey="collection">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" checked={collectionConfig.raw_4k} onChange={(e)=>setCollectionConfig({...collectionConfig, raw_4k: e.target.checked})} disabled={!isAuthenticated} />
                    Collect 4K raw images
                  </label>
                </div>
                <div>
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" checked={collectionConfig.corrected_1080p} onChange={(e)=>setCollectionConfig({...collectionConfig, corrected_1080p: e.target.checked})} disabled={!isAuthenticated} />
                    Collect 1080p fisheye-corrected images
                  </label>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Interval (minutes)</label>
                  <input type="number" min={1} max={60} value={collectionConfig.interval_minutes} onChange={(e)=>setCollectionConfig({...collectionConfig, interval_minutes: Math.max(1, parseInt(e.target.value||'1'))})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
                <div>
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                  <input type="checkbox" checked={collectionConfig.gui_grid} onChange={(e)=>setCollectionConfig({...collectionConfig, gui_grid: e.target.checked})} disabled={!isAuthenticated} />
                    GUI capture grid (11 cams)
                  </label>
                </div>
                <div className="md:col-span-2">
                  <label className="inline-flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" checked={collectionConfig.upload_roboflow} onChange={(e)=>setCollectionConfig({...collectionConfig, upload_roboflow: e.target.checked})} disabled={!isAuthenticated} />
                    Upload all corrected images to Roboflow
                  </label>
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm text-gray-400 mb-1">Output directory</label>
                  <input type="text" value={collectionConfig.output_dir} onChange={(e)=>setCollectionConfig({...collectionConfig, output_dir: e.target.value})} className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2" disabled={!isAuthenticated} />
                </div>
              </div>
            </ConfigSection>
          )}
        </div>
      )}

      {/* Help Text */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h4 className="font-medium mb-2 flex items-center">
          <AlertTriangle className="w-4 h-4 mr-2 text-yellow-400" />
          Configuration Notes
        </h4>
        <ul className="text-sm text-gray-400 space-y-1">
          <li>• Changes require system restart to take effect</li>
          <li>• Invalid JSON will not be saved</li>
          <li>• Backup configurations before making changes</li>
          <li>• Contact admin for production environment changes</li>
        </ul>
      </div>
    </div>
  );
};

export default ConfigurationTab;
