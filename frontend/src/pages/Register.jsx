import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../App";
const T = {
  fr: {
    back: "← Retour",
    title: "Créer un compte",
    sub: "Rejoignez AWAL GPT et commencez à chatter",
    username: "Nom d'utilisateur",
    email: "Email",
    password: "Mot de passe",
    confirm: "Confirmer le mot de passe",
    submit: "Créer un compte →",
    loading: "Création...",
    hasAccount: "Déjà un compte ?",
    signIn: "Se connecter",
    errorMatch: "Les mots de passe ne correspondent pas.",
    errorLength: "Le mot de passe doit faire au moins 6 caractères.",
    defaultError: "Échec de l'inscription.",
  },
  en: {
    back: "← Back",
    title: "Create an account",
    sub: "Join AWAL GPT and start chatting",
    username: "Username",
    email: "Email",
    password: "Password",
    confirm: "Confirm password",
    submit: "Create account →",
    loading: "Creating...",
    hasAccount: "Already have an account?",
    signIn: "Sign in",
    errorMatch: "Passwords do not match.",
    errorLength: "Password must be at least 6 characters.",
    defaultError: "Registration failed.",
  },
  ar: {
    back: "← رجوع",
    title: "إنشاء حساب",
    sub: "انضم إلى AWAL GPT وابدأ المحادثة",
    username: "اسم المستخدم",
    email: "البريد الإلكتروني",
    password: "كلمة المرور",
    confirm: "تأكيد كلمة المرور",
    submit: "إنشاء حساب ←",
    loading: "جارٍ الإنشاء...",
    hasAccount: "لديك حساب بالفعل؟",
    signIn: "تسجيل الدخول",
    errorMatch: "كلمتا المرور غير متطابقتين.",
    errorLength: "يجب أن تتكون كلمة المرور من 6 أحرف على الأقل.",
    defaultError: "فشل التسجيل.",
  },
};
export default function Register() {
  const nav = useNavigate();
  const { register } = useAuth();
  const { language } = useLanguage();
  const t = T[language] || T.fr;
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    confirm: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (form.password !== form.confirm) return setError(t.errorMatch);
    if (form.password.length < 6) return setError(t.errorLength);
    setLoading(true);
    try {
      await register(form.username, form.email, form.password);
      localStorage.setItem("awal_username", form.username);
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
            <label className="authLabel">{t.username}</label>
            <input
              className="authInput"
              type="text"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              required
            />
          </div>
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
          <div className="authField">
            <label className="authLabel">{t.confirm}</label>
            <input
              className="authInput"
              type="password"
              value={form.confirm}
              onChange={(e) => setForm({ ...form, confirm: e.target.value })}
              required
            />
          </div>
          <button className="authSubmit" disabled={loading}>
            {loading ? t.loading : t.submit}
          </button>
        </form>
        <p className="authSwitch">
          {t.hasAccount}{" "}
          <Link to="/login" className="authLink">
            {t.signIn}
          </Link>
        </p>
      </div>
    </div>
  );
}
