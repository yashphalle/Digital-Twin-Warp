import React, { useState, useMemo } from 'react';
import { Package, Grid3x3, Search, Bell, User, TrendingUp, Clock, Activity, Box, Camera, Maximize2 } from 'lucide-react';

// Types
interface Pallet {
  id: string;
  location: string;
  placedAt: Date;
  status: 'active' | 'removed';
  dimensions: {
    width: number;
    height: number;
    depth: number;
  };
  cameraId: number;
}

interface GridCell {
  id: string;
  row: string;
  column: number;
  occupied: boolean;
  palletId?: string;
  cameraZone: number;
}

interface Activity {
  type: 'placed' | 'removed';
  message: string;
  time: string;
}

// Utils
const getCameraZone = (row: string, col: number): number => {
  const rowIndex = row.charCodeAt(0) - 'A'.charCodeAt(0);
  const colIndex = col - 1;
  
  if (rowIndex < 4 && colIndex < 4) return 1;
  if (rowIndex < 4 && colIndex >= 4) return 2;
  if (rowIndex >= 4 && colIndex < 4) return 3;
  if (rowIndex >= 4 && colIndex >= 4) return 4;
  return 1;
};

// Dummy Data Generator
const generateDummyPallets = (): Pallet[] => {
  const locations = ['A1', 'A3', 'B2', 'B4', 'C1', 'C3', 'D2', 'D4', // Zone 1
                     'A5', 'A7', 'B6', 'B8', 'C5', 'C7', 'D6', 'D8', // Zone 2
                     'E1', 'E3', 'F2', 'F4', 'G1', 'G3', 'H2', 'H4', // Zone 3
                     'E5', 'E7', 'F6', 'F8', 'G5', 'G7', 'H6', 'H8']; // Zone 4
  
  return locations.map((loc, index) => {
    const row = loc[0];
    const col = parseInt(loc[1]);
    const cameraId = getCameraZone(row, col);
    
    return {
      id: `PAL-2024-${String(index + 1000).padStart(4, '0')}`,
      location: loc,
      placedAt: new Date(Date.now() - Math.random() * 86400000 * 7),
      status: 'active' as const,
      dimensions: {
        width: 1.2,
        height: 0.15 + Math.random() * 0.3,
        depth: 1.0
      },
      cameraId: cameraId
    };
  });
};

// Generate grid structure
const generateGrid = (): GridCell[] => {
  const rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
  const cols = [1, 2, 3, 4, 5, 6, 7, 8];
  const grid: GridCell[] = [];
  
  rows.forEach(row => {
    cols.forEach(col => {
      grid.push({
        id: `${row}${col}`,
        row,
        column: col,
        occupied: false,
        palletId: undefined,
        cameraZone: getCameraZone(row, col)
      });
    });
  });
  
  return grid;
};

// Header Component
const Header: React.FC = () => {
  return (
    <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <Package className="w-8 h-8 text-blue-500" />
            <h1 className="text-xl font-bold text-white">Warehouse Tracker</h1>
          </div>
        </div>
        
        <div className="flex items-center space-x-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search pallet ID..."
              className="bg-gray-800 text-white pl-10 pr-4 py-2 rounded-lg w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
            <Bell className="w-6 h-6" />
            <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>
          
          <button className="p-2 text-gray-400 hover:text-white transition-colors">
            <User className="w-6 h-6" />
          </button>
        </div>
      </div>
    </header>
  );
};

// Stat Card Component
interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  trend?: number;
  trendLabel?: string;
  color?: 'blue' | 'green' | 'amber';
}

const StatCard: React.FC<StatCardProps> = ({ icon, label, value, trend, trendLabel, color = "blue" }) => {
  const colorClasses = {
    blue: "bg-blue-900/20 text-blue-400 border-blue-800",
    green: "bg-green-900/20 text-green-400 border-green-800",
    amber: "bg-amber-900/20 text-amber-400 border-amber-800"
  };

  return (
    <div className={`rounded-lg p-4 border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="p-2 bg-gray-800 rounded-lg">
          {icon}
        </div>
        {trend && (
          <div className="flex items-center space-x-1 text-sm">
            <TrendingUp className="w-4 h-4" />
            <span>{trend > 0 ? '+' : ''}{trend}</span>
          </div>
        )}
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
      {trendLabel && <div className="text-xs text-gray-500 mt-1">{trendLabel}</div>}
    </div>
  );
};

// Grid Cell Component
interface GridCellProps {
  cell: GridCell;
  pallet?: Pallet;
  onSelect: (cellId: string, pallet?: Pallet) => void;
  selected: string | null;
  showLabel?: boolean;
}

const GridCell: React.FC<GridCellProps> = ({ cell, pallet, onSelect, selected, showLabel = true }) => {
  const isOccupied = !!pallet;
  const isSelected = selected === cell.id;
  
  const cameraColors: Record<number, string> = {
    1: 'border-purple-600 bg-purple-900/20',
    2: 'border-blue-600 bg-blue-900/20',
    3: 'border-green-600 bg-green-900/20',
    4: 'border-amber-600 bg-amber-900/20'
  };
  
  return (
    <div
      onClick={() => onSelect(cell.id, pallet)}
      className={`
        relative aspect-square rounded-md p-1 cursor-pointer transition-all duration-200 border
        ${isOccupied 
          ? isSelected 
            ? 'bg-blue-600 ring-2 ring-blue-400 transform scale-110 z-10' 
            : `${cameraColors[pallet.cameraId]} hover:opacity-80`
          : 'bg-gray-800/50 hover:bg-gray-700/50 border-gray-700'
        }
      `}
    >
      {showLabel && (
        <div className="text-[10px] text-gray-400 absolute top-0.5 left-1">{cell.id}</div>
      )}
      {isOccupied && (
        <div className="flex items-center justify-center h-full">
          <Package className="w-4 h-4 text-white opacity-80" />
        </div>
      )}
    </div>
  );
};

// Camera Feed Component
interface CameraFeedProps {
  cameraId: number;
  onExpand: (cameraId: number) => void;
  palletCount: number;
}

const CameraFeed: React.FC<CameraFeedProps> = ({ cameraId, onExpand, palletCount }) => {
  const cameraColors: Record<number, string> = {
    1: 'border-purple-600',
    2: 'border-blue-600',
    3: 'border-green-600',
    4: 'border-amber-600'
  };
  
  return (
    <div className={`bg-gray-800 rounded-lg border-2 ${cameraColors[cameraId]} overflow-hidden`}>
      <div className="p-2 bg-gray-900 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Camera className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium">Camera {cameraId}</span>
          <span className="text-xs text-gray-500">({palletCount} pallets)</span>
        </div>
        <button
          onClick={() => onExpand(cameraId)}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
        >
          <Maximize2 className="w-4 h-4 text-gray-400" />
        </button>
      </div>
      <div className="p-4 h-32 flex items-center justify-center bg-gray-900/50">
        <div className="text-center">
          <Camera className="w-8 h-8 text-gray-600 mx-auto mb-2" />
          <p className="text-xs text-gray-500">Live Feed</p>
          <p className="text-xs text-gray-600">Zone {cameraId}</p>
        </div>
      </div>
    </div>
  );
};

// Activity Item Component
interface ActivityItemProps {
  activity: Activity;
}

const ActivityItem: React.FC<ActivityItemProps> = ({ activity }) => {
  const getIcon = () => {
    switch (activity.type) {
      case 'placed': return <Package className="w-4 h-4 text-green-400" />;
      case 'removed': return <Package className="w-4 h-4 text-red-400" />;
      default: return <Activity className="w-4 h-4 text-gray-400" />;
    }
  };
  
  return (
    <div className="flex items-start space-x-3 py-3 border-b border-gray-800 last:border-0">
      <div className="p-2 bg-gray-800 rounded-lg">
        {getIcon()}
      </div>
      <div className="flex-1">
        <div className="text-sm text-white">{activity.message}</div>
        <div className="text-xs text-gray-500 mt-1">
          <Clock className="w-3 h-3 inline mr-1" />
          {activity.time}
        </div>
      </div>
    </div>
  );
};

// Main App Component
export default function App() {
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [selectedPallet, setSelectedPallet] = useState<Pallet | undefined>(undefined);
  const [expandedCamera, setExpandedCamera] = useState<number | null>(null);
  
  // Generate dummy data
  const pallets = useMemo(() => generateDummyPallets(), []);
  const grid = useMemo(() => {
    const baseGrid = generateGrid();
    // Map pallets to grid
    pallets.forEach(pallet => {
      const cell = baseGrid.find(c => c.id === pallet.location);
      if (cell) {
        cell.occupied = true;
        cell.palletId = pallet.id;
      }
    });
    return baseGrid;
  }, [pallets]);
  
  // Calculate pallets per camera
  const palletsPerCamera = useMemo(() => {
    const counts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };
    pallets.forEach(pallet => {
      counts[pallet.cameraId]++;
    });
    return counts;
  }, [pallets]);
  
  // Dummy activities
  const activities: Activity[] = [
    { type: 'placed', message: 'Pallet PAL-2024-1012 placed at B4', time: '5 minutes ago' },
    { type: 'removed', message: 'Pallet PAL-2024-1008 removed from A3', time: '12 minutes ago' },
    { type: 'placed', message: 'Pallet PAL-2024-1011 placed at C5', time: '28 minutes ago' },
  ];
  
  const handleCellSelect = (cellId: string, pallet?: Pallet) => {
    setSelectedCell(cellId);
    setSelectedPallet(pallet);
  };
  
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };
  
  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <Header />
      
      <div className="flex flex-1 overflow-hidden">
        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Global Grid Map */}
          <div className="flex-1 p-6">
            <div className="bg-gray-800 rounded-lg p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold">Global Warehouse Map</h2>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-purple-600 rounded"></div>
                    <span className="text-xs text-gray-400">Zone 1</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-blue-600 rounded"></div>
                    <span className="text-xs text-gray-400">Zone 2</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-600 rounded"></div>
                    <span className="text-xs text-gray-400">Zone 3</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-amber-600 rounded"></div>
                    <span className="text-xs text-gray-400">Zone 4</span>
                  </div>
                </div>
              </div>
              
              <div className="flex-1 flex items-center justify-center">
                <div className="grid grid-cols-8 gap-1.5 p-4 bg-gray-900 rounded-lg">
                  {grid.map(cell => {
                    const pallet = pallets.find(p => p.location === cell.id);
                    return (
                      <GridCell
                        key={cell.id}
                        cell={cell}
                        pallet={pallet}
                        onSelect={handleCellSelect}
                        selected={selectedCell}
                      />
                    );
                  })}
                </div>
              </div>
              
              {/* Grid Labels */}
              <div className="mt-4 text-center text-xs text-gray-500">
                Rows: A-H (Top to Bottom) | Columns: 1-8 (Left to Right)
              </div>
            </div>
          </div>
          
          {/* Camera Feeds */}
          <div className="p-6 pt-0">
            <div className="grid grid-cols-4 gap-4">
              {[1, 2, 3, 4].map(cameraId => (
                <CameraFeed
                  key={cameraId}
                  cameraId={cameraId}
                  onExpand={setExpandedCamera}
                  palletCount={palletsPerCamera[cameraId]}
                />
              ))}
            </div>
          </div>
        </div>
        
        {/* Right Sidebar */}
        <div className="w-96 bg-gray-800 p-6 border-l border-gray-700 overflow-y-auto">
          <div className="space-y-6">
            {/* Stats */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Overview</h3>
              <div className="space-y-3">
                <StatCard
                  icon={<Package className="w-5 h-5" />}
                  label="Total Active Pallets"
                  value={pallets.length}
                  trend={3}
                  trendLabel="vs yesterday"
                  color="blue"
                />
                <StatCard
                  icon={<TrendingUp className="w-5 h-5" />}
                  label="Today's Throughput"
                  value="23"
                  trend={5}
                  color="green"
                />
                <StatCard
                  icon={<Grid3x3 className="w-5 h-5" />}
                  label="Space Utilization"
                  value={`${Math.round((pallets.length / 64) * 100)}%`}
                  color="amber"
                />
              </div>
            </div>
            
            {/* Camera Zone Stats */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Zone Distribution</h3>
              <div className="bg-gray-900 rounded-lg p-4 space-y-3">
                {[1, 2, 3, 4].map(zone => (
                  <div key={zone} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Camera className="w-4 h-4 text-gray-400" />
                      <span className="text-sm">Zone {zone}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold">{palletsPerCamera[zone]}</span>
                      <span className="text-xs text-gray-500">pallets</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Selected Pallet Info */}
            {selectedPallet && (
              <div>
                <h3 className="text-lg font-semibold mb-4">Pallet Details</h3>
                <div className="bg-gray-900 rounded-lg p-4 space-y-3">
                  <div>
                    <div className="text-sm text-gray-400">Pallet ID</div>
                    <div className="font-mono">{selectedPallet.id}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Location</div>
                    <div className="text-xl font-semibold">{selectedPallet.location}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Camera Zone</div>
                    <div>Zone {selectedPallet.cameraId}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Placed At</div>
                    <div>{formatDate(selectedPallet.placedAt)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">Duration</div>
                    <div className="text-blue-400">
                      {Math.floor((Date.now() - selectedPallet.placedAt.getTime()) / (1000 * 60 * 60))} hours
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Activity Feed */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
              <div className="bg-gray-900 rounded-lg p-4">
                {activities.map((activity, index) => (
                  <ActivityItem key={index} activity={activity} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}