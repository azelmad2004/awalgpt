import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { useLanguage } from '../App';
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const api = axios.create({ baseURL: API_URL, headers: { 'Content-Type': 'application/json' } });
api.interceptors.request.use(config => {
  const t = localStorage.getItem('awal_token');
  if (t) config.headers.Authorization = `Bearer ${t}`;
  return config;
});
api.interceptors.response.use(
  res => res,
  err => {
    if (err?.response?.status === 401) {
      localStorage.removeItem('awal_token');
      localStorage.removeItem('awal_username');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

const T = {
  fr: {
    newChat: 'Nouveau chat', today: "Aujourd'hui", days30: '30 derniers jours',
    settings: 'Paramètres', signOut: 'Déconnexion',
    welcomeSub: 'Votre assistant intelligent pour la langue et la culture amazighe.',
    placeholder: 'Pose ta question en amazigh, français ou darija...',
    hint: 'Entrée pour envoyer · Shift+Entrée pour nouvelle ligne',
    selectDomain: 'Choisis un domaine pour commencer',
    changeDomain: 'Changer',
  },
  en: {
    newChat: 'New chat', today: 'Today', days30: '30 days',
    settings: 'Settings', signOut: 'Sign out',
    welcomeSub: 'Your smart assistant for the Amazigh language and culture.',
    placeholder: 'Ask in Amazigh, French or Darija...',
    hint: 'Enter to send · Shift+Enter for new line',
    selectDomain: 'Choose a domain to get started',
    changeDomain: 'Change',
  },
  ar: {
    newChat: 'محادثة جديدة', today: 'اليوم', days30: 'آخر 30 يومًا',
    settings: 'الإعدادات', signOut: 'تسجيل الخروج',
    welcomeSub: 'مساعدك الذكي للغة والثقافة الأمازيغية.',
    placeholder: 'اسأل بالأمازيغية أو الفرنسية أو الدارجة...',
    hint: 'Enter للإرسال · Shift+Enter لسطر جديد',
    selectDomain: 'اختر مجالاً للبدء',
    changeDomain: 'تغيير',
  },
};
const DOMAINS = {
  fr: [
    { id: 'health',    label: 'Santé',       desc: 'Vocabulaire médical amazigh' },
    { id: 'economy',   label: 'Économie',    desc: 'Commerce et finances en amazigh' },
    { id: 'education', label: 'Éducation',   desc: 'Apprentissage et enseignement' },
    { id: 'culture',   label: 'Culture',     desc: 'Traditions et patrimoine amazigh' },
    { id: 'tech',      label: 'Technologie', desc: 'Terminologie tech en tamazight' },
    { id: 'daily',     label: 'Quotidien',   desc: 'Conversations de tous les jours' },
  ],
  en: [
    { id: 'health',    label: 'Health',      desc: 'Medical vocabulary in Amazigh' },
    { id: 'economy',   label: 'Economy',     desc: 'Commerce and finance in Amazigh' },
    { id: 'education', label: 'Education',   desc: 'Learning and teaching' },
    { id: 'culture',   label: 'Culture',     desc: 'Amazigh traditions and heritage' },
    { id: 'tech',      label: 'Technology',  desc: 'Tech terminology in Tamazight' },
    { id: 'daily',     label: 'Daily life',  desc: 'Everyday conversations' },
  ],
  ar: [
    { id: 'health',    label: 'الصحة',         desc: 'المفردات الطبية بالأمازيغية' },
    { id: 'economy',   label: 'الاقتصاد',      desc: 'التجارة والمالية بالأمازيغية' },
    { id: 'education', label: 'التعليم',        desc: 'التعلم والتدريس' },
    { id: 'culture',   label: 'الثقافة',        desc: 'التقاليد والتراث الأمازيغي' },
    { id: 'tech',      label: 'التكنولوجيا',    desc: 'المصطلحات التقنية بالتامازيغت' },
    { id: 'daily',     label: 'الحياة اليومية', desc: 'محادثات يومية' },
  ],
};

export default function Chat() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const { language } = useLanguage();
  const [showSettings, setShowSettings]   = useState(false);
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId]   = useState(null);
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState('');
  const [loading, setLoading]             = useState(false);
  const [sidebarOpen, setSidebarOpen]     = useState(true);
  const [activeDomain, setActiveDomain]   = useState(null);
  const [activeModel, setActiveModel]     = useState('groq');
  const [userInfo, setUserInfo]           = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef    = useRef(null);
  const t       = T[language];
  const domains = DOMAINS[language];
  useEffect(() => {
    const fetchMe = async () => {
      try {
        const r = await api.get('/auth/me');
        setUserInfo(r.data);
      } catch {
        if (!user) nav('/login');
      }
    };
    fetchMe();
  }, []);
  const displayName = (() => {
    if (userInfo?.username?.trim()) return userInfo.username.trim();
    if (userInfo?.email)            return userInfo.email.split('@')[0];
    if (user?.username?.trim())     return user.username.trim();
    if (user?.email)                return user.email.split('@')[0];
    const stored = localStorage.getItem('awal_username');
    if (stored?.trim())             return stored.trim();
    return 'Guest';
  })();
  const displayEmail = userInfo?.email || user?.email || '';
  const getGreeting = () => {
    const h = new Date().getHours();
    if (h < 12) return language === 'fr' ? 'Bonjour' : language === 'ar' ? 'صباح الخير' : 'Good morning';
    if (h < 18) return language === 'fr' ? 'Bon après-midi' : language === 'ar' ? 'مساء الخير' : 'Good afternoon';
    return language === 'fr' ? 'Bonsoir' : language === 'ar' ? 'مساء الخير' : 'Good evening';
  };
  useEffect(() => { loadConversations(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);
  const loadConversations = useCallback(async () => {
    try {
      const r = await api.get('/conversations');
      setConversations(r.data.conversations || []);
    } catch (err) { console.error(err); }
  }, []);
  const loadConversation = async (convId) => {
    try {
      const r = await api.get(`/conversations/${convId}/messages`);
      setMessages(r.data.messages.map(m => ({ role: m.sender === 'user' ? 'user' : 'bot', content: m.content })));
      setActiveConvId(convId);
    } catch (err) { console.error(err); }
  };
  const deleteConversation = async (id, e) => {
    e.stopPropagation();
    try {
      await api.delete(`/conversations/${id}`);
      setConversations(prev => prev.filter(c => c.id !== id));
      if (activeConvId === id) { setActiveConvId(null); setMessages([]); }
    } catch (err) { console.error(err); }
  };
  const newConversation = () => {
    setActiveConvId(null); setMessages([]); setActiveDomain(null);
    textareaRef.current?.focus();
  };
  const handleInputChange = e => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };
  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };
const STEP_LABELS = {
  intent:   { icon: '🔍', text: 'Analyse de la question...' },
  rag:      { icon: '📚', text: 'Recherche Tamazight...' },
  rag_done: { icon: '📚', text: null }, 
  llm:      { icon: '🧠', text: 'Génération de la réponse...' },
  vocab:    { icon: '✅', text: 'Vérification vocabulaire...' },
};
const [thinkingSteps, setThinkingSteps] = useState([]); 
const [streamStarted, setStreamStarted] = useState(false); 
const sendMessage = useCallback(async () => {
  const text = input.trim();
  if (!text || loading) return;
  setMessages(prev => [...prev, { role: 'user', content: text }]);
  setInput('');
  if (textareaRef.current) textareaRef.current.style.height = 'auto';
  setLoading(true);
  setThinkingSteps([]);
  setStreamStarted(false);
  try {
    const token = localStorage.getItem('awal_token');
    const response = await fetch(`${API_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        message: text,
        conversation_id: activeConvId,
        domain: activeDomain || null,
        model: activeModel,
      })
    });
    if (!response.ok) throw new Error('Stream request failed');
    const headerConvId = response.headers.get('X-Conversation-Id');
    if (headerConvId && !activeConvId) {
      setActiveConvId(headerConvId);
      loadConversations();
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let streamedContent = '';
    let messageAdded = false;
    let sseBuffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });
      const parts = sseBuffer.split('\n\n');
      sseBuffer = parts.pop(); 
      for (const msg of parts) {
        if (!msg.trim()) continue;
        let eventType = 'token';
        let dataLine  = '';
        for (const line of msg.split('\n')) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          if (line.startsWith('data: '))  dataLine  = line.slice(6);
        }
        if (eventType === 'step') {
          try {
            const ev = JSON.parse(dataLine);
            setThinkingSteps(prev => {
              const updated = prev.map(s => ({ ...s, done: true }));
              if (updated.some(s => s.step === ev.step)) return updated;
              return [...updated, { step: ev.step, label: ev.label, done: false }];
            });
          } catch {}
        } else if (eventType === 'token') {
          try {
            const text = JSON.parse(dataLine);
            streamedContent += text;
            setStreamStarted(true);
            if (!messageAdded) {
              setMessages(prev => [...prev, { role: 'bot', content: streamedContent }]);
              messageAdded = true;
            } else {
              setMessages(prev => {
                const n = [...prev];
                n[n.length - 1].content = streamedContent;
                return n;
              });
            }
          } catch {}
        }
      }
    }
  } catch (err) {
    console.error(err);
    setMessages(prev => {
      const next = [...prev];
      if (next[next.length - 1]?.role !== 'bot') {
        return [...next, { role: 'bot', content: 'Erreur de connexion. Veuillez réessayer.' }];
      }
      next[next.length - 1].content = 'Erreur de connexion. Veuillez réessayer.';
      return next;
    });
  } finally {
    setLoading(false);
    setThinkingSteps([]);
    setStreamStarted(false);
  }
}, [input, loading, activeConvId, activeDomain, activeModel, loadConversations]);
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // On pourrait ajouter un toast ici si nécessaire
  };
  const now     = new Date();
  const grouped = conversations.reduce((acc, c) => {
    const d    = new Date(c.updated_at || c.created_at);
    const diff = (now - d) / 86400000;
    if (diff < 1) acc.today.push(c);
    else if (diff < 30) acc.days30.push(c);
    return acc;
  }, { today: [], days30: [] });
  const activeDomainObj = domains.find(d => d.id === activeDomain);
  return (
    <div className="layout">
      <style>{`
        .thinkingSteps { display: flex; flex-direction: column; gap: 6px; padding: 4px 0; }
        .thinkingStep  { display: flex; align-items: center; gap: 8px; font-size: 13px; transition: opacity 0.3s; }
        .thinkingStep.stepDone  { opacity: 0.45; }
        .thinkingStep.stepActive { opacity: 1; font-weight: 500; }
        .stepIcon { width: 18px; text-align: center; font-size: 14px; flex-shrink: 0; }
        .stepDone .stepIcon { color: #22c55e; }
        .stepLabel { color: var(--text-secondary, #6b7280); }
        .stepActive .stepLabel { color: var(--text-primary, #111827); }
        .stepSpinner {
          display: inline-block; width: 12px; height: 12px;
          border: 2px solid #d1d5db; border-top-color: #6366f1;
          border-radius: 50%; animation: spinStep 0.7s linear infinite;
        }
        .messageActions {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-top: 10px;
          padding-top: 8px;
          border-top: 1px solid rgba(0,0,0,0.03);
        }
        .actionIcon {
          background: none;
          border: none;
          padding: 4px;
          color: var(--red);
          opacity: 0.6;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 6px;
        }
        .actionIcon:hover {
          opacity: 1;
          background: rgba(220, 38, 38, 0.08);
          transform: scale(1.1);
        }
        @keyframes spinStep { to { transform: rotate(360deg); } }
      `}</style>
      {}
      {showSettings && (
        <div className="modalOverlay" onClick={() => setShowSettings(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modalHeader">
              <h3>{t.settings}</h3>
              <button className="closeModal" onClick={() => setShowSettings(false)}>×</button>
            </div>
            <div className="modalBody">
              <p style={{ fontSize: 13, color: '#6b7280' }}>
                {language === 'fr' ? "La langue est définie depuis la page d'accueil."
                  : language === 'ar' ? 'يتم تحديد اللغة من الصفحة الرئيسية.'
                  : 'Language is set from the home page.'}
              </p>
            </div>
          </div>
        </div>
      )}
      {}
      {sidebarOpen && <div className="sidebarOverlay" onClick={() => setSidebarOpen(false)}></div>}
      {}
      <aside className={`sidebar ${sidebarOpen ? 'sideOpen' : ''}`}>
        <div className="sideHeader">
          <button className="newChatBtn" onClick={newConversation}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            {t.newChat}
          </button>
        </div>
        <div className="sideContent">
          {grouped.today.length > 0 && (
            <div className="dateGroup">
              <div className="dateLabel">{t.today}</div>
              {grouped.today.map(c => (
                <div key={c.id} className={`convItem ${activeConvId === c.id ? 'convActive' : ''}`} onClick={() => loadConversation(c.id)}>
                  <svg className="convIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  <span className="convTitle">{c.title || 'Conversation'}</span>
                  <button onClick={e => deleteConversation(c.id, e)} className="delBtn">×</button>
                </div>
              ))}
            </div>
          )}
          {grouped.days30.length > 0 && (
            <div className="dateGroup">
              <div className="dateLabel">{t.days30}</div>
              {grouped.days30.map(c => (
                <div key={c.id} className={`convItem ${activeConvId === c.id ? 'convActive' : ''}`} onClick={() => loadConversation(c.id)}>
                  <svg className="convIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  <span className="convTitle">{c.title || 'Conversation'}</span>
                  <button onClick={e => deleteConversation(c.id, e)} className="delBtn">×</button>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="sideFooter">
          <button className="userBtn">
            <div className="userAvatar">{displayName[0]?.toUpperCase()}</div>
            <div className="userInfo">
              <div className="userName">{displayName}</div>
              <div className="userEmail">{displayEmail}</div>
            </div>
          </button>
          <button className="signOutBtn" onClick={() => { logout(); nav('/'); }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            {t.signOut}
          </button>
        </div>
      </aside>
      {}
      <main className="main">
        <div className="mainHeader">
          <div className="headerLeft">
            <button className="menuBtn" onClick={() => setSidebarOpen(v => !v)}>☰</button>
            <div className="headerLogo">
              <img src="/logo.png" alt="Logo" width={26} />
              <span>AWAL GPT</span>
            </div>
            {activeDomainObj && (
              <div className="headerDomainBadge">
                <span>{activeDomainObj.label}</span>
                <button onClick={() => setActiveDomain(null)}>×</button>
              </div>
            )}
          </div>
          <div className="headerActions">
            <button className="settingsBtn" onClick={() => setShowSettings(true)} title={t.settings}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
          </div>
        </div>
        <div className="messagesArea">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="welcome">
                <div className="greetingRow">
                  <img src="/logo.png" alt="Logo" className="welcomeLogo" />
                  <h1 className="welcomeTitle">
                    {getGreeting()}, <span className="greetingName">{displayName}</span>
                  </h1>
                </div>
                <p className="welcomeSub">{t.welcomeSub}</p>
                {!activeDomain ? (
                  <div className="domainSection">
                    <p className="domainSectionLabel">{t.selectDomain}</p>
                    <div className="domainGrid">
                      {domains.map(d => (
                        <button key={d.id} className="domainCard" onClick={() => setActiveDomain(d.id)}>
                          <span className="domainCardLabel">{d.label}</span>
                          <span className="domainCardDesc">{d.desc}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="activeDomainWelcome">
                    <div>
                      <div className="activeDomainName">{activeDomainObj?.label}</div>
                      <div className="activeDomainDesc">{activeDomainObj?.desc}</div>
                    </div>
                    <button className="changeDomainBtn" onClick={() => setActiveDomain(null)}>
                      {t.changeDomain}
                    </button>
                  </div>
                )}
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role === 'user' ? 'messageUser' : 'messageBot'}`}>
                  <div className="messageAvatar">
                    {msg.role === 'user'
                      ? <div className="userAvatarMsg">{displayName[0]?.toUpperCase()}</div>
                      : <img src="/logo.png" className="botAvatarImg" alt="Logo" />}
                  </div>
                  <div className="messageContent">
                    <div className="messageName">
                      {msg.role === 'user' ? displayName : 'AWAL GPT'}
                      {msg.role === 'user' && activeDomainObj && (
                        <span className="msgDomainTag">{activeDomainObj.label}</span>
                      )}
                    </div>
                  <div className="messageText">
                    {msg.role === 'user' ? (
                      msg.content
                    ) : (() => {
                      const processed = msg.content
                        .replace(/\[TAM\]|\[AR\]/g, '')
                        .replace(/\[TAM(?=[a-z\s])/g, '')
                        .replace(/\[AR(?=[a-z\s])/g, '')
                        .replace(/\[TAM$/, '')
                        .replace(/\[AR$/, '')
                        .trim();
                      return (
                        <>
                          <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                            table: ({node, ...props}) => <table style={{ borderCollapse: 'collapse', width: '100%', margin: '12px 0', fontSize: 14 }} {...props} />,
                            th: ({node, ...props}) => <th style={{ background: '#f3f4f6', border: '1px solid #e5e7eb', padding: '8px 12px', textAlign: 'left', fontWeight: 600 }} {...props} />,
                            td: ({node, ...props}) => <td style={{ border: '1px solid #e5e7eb', padding: '8px 12px' }} {...props} />,
                            tr: ({node, ...props}) => <tr style={{ borderBottom: '1px solid #e5e7eb' }} {...props} />,
                            ol: ({node, ...props}) => <ol style={{ paddingLeft: 20, margin: '8px 0', lineHeight: 1.8 }} {...props} />,
                            ul: ({node, ...props}) => <ul style={{ paddingLeft: 20, margin: '8px 0', lineHeight: 1.8 }} {...props} />,
                            li: ({node, ...props}) => <li style={{ marginBottom: 4 }} {...props} />,
                          }}>
                            {processed}
                          </ReactMarkdown>
                          
                          <div className="messageActions">
                            <button className="actionIcon" onClick={() => copyToClipboard(processed)} title="Copier">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                              </svg>
                            </button>
                            <button className="actionIcon" title="J'aime">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                              </svg>
                            </button>
                            <button className="actionIcon" title="Je n'aime pas">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
                              </svg>
                            </button>
                            <button className="actionIcon" onClick={() => sendMessage(messages[idx-1]?.content)} title="Régénérer">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="23 4 23 10 17 10"></polyline>
                                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                              </svg>
                            </button>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                 </div>
                </div>
              ))
            )}
            {loading && !streamStarted && (
              <div className="message messageBot">
                <div className="messageAvatar">
                  <img src="/logo.png" className="botAvatarImg" alt="Logo" />
                </div>
                <div className="messageContent">
                  <div className="messageName">AWAL GPT</div>
                  <div className="thinkingBlock">
                    {thinkingSteps.length === 0 ? (
                      <div className="typing"><span /><span /><span /></div>
                    ) : (
                      <div className="thinkingSteps">
                        {thinkingSteps.map((s, i) => (
                          <div key={i} className={`thinkingStep ${s.done ? 'stepDone' : 'stepActive'}`}>
                            <span className="stepIcon">{s.done ? '✓' : <span className="stepSpinner" />}</span>
                            <span className="stepLabel">{s.label}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
        {}
        <div className="inputArea">
          <div className="inputContainer">
            {activeDomainObj && (
              <div className="inputDomainBar">
                <span className="inputDomainLabel">{activeDomainObj.label}</span>
                <span className="inputDomainAuto">
                  {language === 'fr' ? '— domaine actif' : language === 'ar' ? '— المجال النشط' : '— active domain'}
                </span>
                <button className="inputDomainClose" onClick={() => setActiveDomain(null)}>×</button>
              </div>
            )}
            <div className="inputBox">
              <textarea
                ref={textareaRef}
                className="textarea"
                placeholder={t.placeholder}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={loading}
              />
              <button className="sendBtn" onClick={sendMessage} disabled={loading || !input.trim()}>
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}