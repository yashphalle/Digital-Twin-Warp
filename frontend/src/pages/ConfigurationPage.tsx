import React from 'react';
import ConfigurationTab from '../components/ConfigurationTab';
import { useAuth } from '../contexts/AuthContext';

const ConfigurationPage: React.FC = () => {
  const { isAuthenticated } = useAuth();
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <img src="/logo3.png" alt="WARP Logo" className="h-10 w-auto" onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }} />
            <h1 className="text-xl font-semibold">Configuration</h1>
          </div>
          <a href="/demo" className="text-sm text-blue-400 hover:text-blue-300">← Back to Portal</a>
        </div>
      </header>
      <main className="max-w-7xl mx-auto p-6">
        <ConfigurationTab isAuthenticated={isAuthenticated} />
      </main>
    </div>
  );
};

export default ConfigurationPage;
