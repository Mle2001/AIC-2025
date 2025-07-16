// src/App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Layout/Sidebar';
import Dashboard from './components/Admin/Dashboard';
import AdminDashboard from './components/Admin/AdminDashboard';
import { Chat } from './components/Chat';
import { VideoRAGDashboard } from './components/Admin/VideoRAG';
import './App.css';
import './components/Admin/VideoRAG/VideoRAG.css';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex' }}>
        <Sidebar />
        <Routes>
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/chat" element={<Chat showVideoSelector={true} />} />
          <Route path="/videorag" element={<VideoRAGDashboard />} />
          <Route path="/" element={<AdminDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;