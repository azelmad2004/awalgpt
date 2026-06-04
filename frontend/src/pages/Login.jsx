import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../App";
const T = {
  fr: {
    back: "← Retour",
    title: "Bon retour !",
    sub: "Connectez-vous pour continuer",
    email: "Email",
    password: "Mot de passe",
    submit: "Se connecter →",
    loading: "Connexion...",
    noAccount: "Pas encore de compte ?",
    createAccount: "Créer un compte",
    defaultError: "Email ou mot de passe invalide.",
  },
  en: {
    back: "← Back",
    title: "Welcome back!",
    sub: "Sign in to continue",
    email: "Email",
    password: "Password",
    submit: "Sign in →",
    loading: "Signing in...",
    noAccount: "No account yet?",
    createAccount: "Create an account",
    defaultError: "Invalid email or password.",
  },
  ar: {
    back: "← رجوع",
    title: "مرحباً بعودتك!",
    sub: "سجّل دخولك للمتابعة",
    email: "البريد الإلكتروني",
    password: "كلمة المرور",
    submit: "تسجيل الدخول ←",
    loading: "جارٍ التسجيل...",
    noAccount: "ليس لديك حساب؟",
    createAccount: "إنشاء حساب",
    defaultError: "بريد إلكتروني أو كلمة مرور غير صحيحة.",
  },
};
export default function Login() {
  const nav = useNavigate();
  const { login } = useAuth();
  const { language } = useLanguage();
  const t = T[language] || T.fr;
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(form.email, form.password);
      nav("/chat");
    } catch (err) {
      setError(err.response?.data?.detail || t.defaultError);
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="authPage">
      <div className="authBg1" />
      <div className="authBg2" />
      <button onClick={() => nav("/")} className="authBack">
        {t.back}
      </button>
      <div className="authCard">
        <div className="authCardLogo">
          <img src="/logo.png" alt="Logo" style={{ width: 40 }} />
        </div>
        <h1 className="authTitle">{t.title}</h1>
        <p className="authSub">{t.sub}</p>
        {error && <div className="authErr">{error}</div>}
        <form onSubmit={onSubmit} className="authForm">
          <div className="authField">
            <label className="authLabel">{t.email}</label>
            <input
              className="authInput"
              type="email"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              required
            />
          </div>
          <div className="authField">
            <label className="authLabel">{t.password}</label>
            <input
              className="authInput"
              type="password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
            />
          </div>
          <button className="authSubmit" disabled={loading}>
            {loading ? t.loading : t.submit}
          </button>
        </form>
        <p className="authSwitch">
          {t.noAccount}{" "}
          <Link to="/register" className="authLink">
            {t.createAccount}
          </Link>
        </p>
      </div>
    </div>
  );
}
