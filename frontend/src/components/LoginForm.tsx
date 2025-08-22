import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Lock, ArrowLeft, AlertCircle } from 'lucide-react';

interface LoginFormProps {
  onLogin: (isAuthenticated: boolean) => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLogin }) => {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Simple frontend authentication (hardcoded)
    if (username === 'admin' && password === '1234') {
      onLogin(true);
      navigate('/admin');
    } else {
      setError('Invalid credentials. Please try again.');
      onLogin(false);
    }
    
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-900/10 to-purple-900/10"></div>
      
      <div className="relative w-full max-w-md px-6">
        {/* Back Button */}
        <button
          onClick={() => navigate('/')}
          className="flex items-center text-gray-400 hover:text-white mb-8 transition-colors duration-200"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </button>

        {/* Login Card */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <img src="/logo3.png" alt="WARP Logo" className="w-16 h-12" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Digital Twin
              </h1>
            </div>
            <h2 className="text-xl font-semibold text-gray-300">Admin Login</h2>
            <p className="text-gray-500 text-sm mt-2">Access warehouse configuration and management</p>
          </div>

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Username Field */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  placeholder="Enter username"
                  required
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                  placeholder="Enter password"
                  required
                />
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center space-x-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg p-3">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">{error}</span>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Signing In...
                </div>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Demo Access */}
          <div className="mt-6 pt-6 border-t border-gray-800">
            <p className="text-center text-gray-500 text-sm mb-3">
              Want to explore without logging in?
            </p>
            <button
              onClick={() => navigate('/demo')}
              className="w-full border border-gray-600 hover:border-blue-500 hover:bg-blue-500/10 text-gray-300 hover:text-white font-medium py-2 px-4 rounded-lg transition-all duration-200"
            >
              Continue as Guest
            </button>
          </div>
        </div>

        {/* Help Text */}
        <div className="text-center mt-6">
          <p className="text-gray-500 text-xs">
            For demo purposes: Username: admin, Password: 1234
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;
