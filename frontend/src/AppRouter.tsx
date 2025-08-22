import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import LandingPage from './components/LandingPage';
import LoginForm from './components/LoginForm';
import App from './App';
import ConfigurationPage from './pages/ConfigurationPage';

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  return <>{children}</>;
};

// Admin Route Component (authenticated app)
const AdminRoute: React.FC = () => {
  const { isAuthenticated } = useAuth();
  return <App isAuthenticated={isAuthenticated} />; // Route to original App shell
};

// Demo Route Component (guest app)
const DemoRoute: React.FC = () => {
  return <App isAuthenticated={false} />; // Route to original App shell for guest
};

// Login Route Component
const LoginRoute: React.FC = () => {
  const { login } = useAuth();

  return (
    <LoginForm onLogin={(authenticated) => {
      if (authenticated) {
        login();
      }
    }} />
  );
};

// Main App Router
const AppRouter: React.FC = () => {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Landing Page */}
          <Route path="/" element={<LandingPage />} />

          {/* Login Page */}
          <Route path="/login" element={<LoginRoute />} />

          {/* Admin Portal */}
          <Route path="/admin" element={<AdminRoute />} />

          {/* Guest/Demo Portal */}
          <Route path="/demo" element={<DemoRoute />} />

          {/* Config page (same UI, guest writes blocked) */}
          <Route path="/config" element={<ConfigurationPage />} />

          {/* Redirect unknown routes to landing */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
};

export default AppRouter;
