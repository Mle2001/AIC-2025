// src/hook/useAuth.js
import { useContext } from 'react';

// Nếu bạn đã có AuthContext, import nó ở đây
// import { AuthContext } from '../context/AuthContext';

export function useAuth() {
  // Nếu dùng context thực tế, thay thế dòng dưới bằng: return useContext(AuthContext);
  // Dưới đây là mock cho dev/test/demo:
  return { user: { id: 'user1', name: 'Demo User' } };
}