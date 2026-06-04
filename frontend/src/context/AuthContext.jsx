import React, { createContext, useContext, useState, useEffect } from "react";
import axios from "axios";
const AuthContext = createContext(null);
axios.defaults.baseURL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const token = localStorage.getItem("awal_token");
    const saved = localStorage.getItem("awal_user");
    if (token && saved) {
      try {
        setUser(JSON.parse(saved));
        axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
      } catch {
        localStorage.removeItem("awal_token");
        localStorage.removeItem("awal_user");
      }
    }
    setLoading(false);
  }, []);
  const _save = (token, u) => {
    localStorage.setItem("awal_token", token);
    localStorage.setItem("awal_user", JSON.stringify(u));
    axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
    setUser(u);
  };
  const register = async (username, email, password) => {
    const r = await axios.post("/auth/register", { username, email, password });
    _save(r.data.token, r.data.user);
    return r.data.user;
  };
  const login = async (email, password) => {
    const r = await axios.post("/auth/login", { email, password });
    _save(r.data.token, r.data.user);
    return r.data.user;
  };
  const logout = () => {
    localStorage.removeItem("awal_token");
    localStorage.removeItem("awal_user");
    delete axios.defaults.headers.common["Authorization"];
    setUser(null);
  };
  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
export function useAuth() {
  return useContext(AuthContext);
}
