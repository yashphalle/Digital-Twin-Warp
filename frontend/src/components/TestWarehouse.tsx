import React from 'react';

const TestWarehouse: React.FC = () => {
  return (
    <div className="p-6 bg-gray-800 rounded-lg h-full">
      <h2 className="text-white text-xl mb-4">Test Warehouse View</h2>
      
      <div className="flex justify-center items-center" style={{ minHeight: '400px' }}>
        <div className="relative">
          {/* Warehouse boundary */}
          <div
            className="relative bg-gray-100 border-2 border-gray-600"
            style={{
              width: '600px',
              height: '400px'
            }}
          >
            {/* Test object markers */}
            <div
              className="absolute w-3 h-3 bg-red-500 rounded-full"
              style={{
                left: '20%',
                top: '30%',
                transform: 'translate(-50%, -50%)'
              }}
            />
            <div
              className="absolute w-3 h-3 bg-blue-500 rounded-full"
              style={{
                left: '60%',
                top: '70%',
                transform: 'translate(-50%, -50%)'
              }}
            />
            <div
              className="absolute w-3 h-3 bg-green-500 rounded-full"
              style={{
                left: '80%',
                top: '20%',
                transform: 'translate(-50%, -50%)'
              }}
            />
            
            {/* Center text */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-gray-600 text-center">
                <div className="text-lg font-bold">Test Warehouse</div>
                <div className="text-sm">10m × 8m</div>
              </div>
            </div>
          </div>
          
          {/* Dimension labels */}
          <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-gray-400 text-sm">
            Width: 10m
          </div>
          <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -rotate-90 text-gray-400 text-sm">
            Length: 8m
          </div>
        </div>
      </div>
      
      <div className="mt-4 text-gray-400 text-sm text-center">
        Test warehouse layout with sample objects
      </div>
    </div>
  );
};

export default TestWarehouse;
