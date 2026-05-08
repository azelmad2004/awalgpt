import React, { useState, createContext, useContext, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Landing from "./pages/Landing";
import Chat from "./pages/Chat";
import Login from "./pages/Login";
import Register from "./pages/Register";
import "./styles.css";
export const LanguageContext = createContext();
export function useLanguage() {
  return useContext(LanguageContext);
}
export default function App() {
  const [language, setLanguageState] = useState(
    () => localStorage.getItem("awal_lang") || "fr",
  );
  const setLanguage = (lang) => {
    localStorage.setItem("awal_lang", lang);
    setLanguageState(lang);
  };
  useEffect(() => {
    document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
  }, [language]);
  return (
    <LanguageContext.Provider value={{ language, setLanguage }}>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </LanguageContext.Provider>
  );
}
