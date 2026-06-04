import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../App";
const T = {
  fr: {
    features: "Fonctionnalités",
    process: "Processus",
    testimonials: "Témoignages",
    login: "Connexion",
    startFree: "Commencer gratuitement →",
    contact: "Contactez-nous",
    heroTitle: "Parle avec l'IA en",
    heroRed: "Tamazight latinisé",
    heroSub:
      "Discute, apprends et explore la langue amazighe grâce à un assistant IA intelligent. Pose tes questions en Amazigh latinisé — notre IA comprend et répond.",
    feat1T: "Conversation naturelle",
    feat1D:
      "Échange librement en amazigh latinisé. L'IA s'adapte à ta façon d'écrire.",
    feat2T: "Apprentissage de la langue",
    feat2D:
      "Apprends le vocabulaire, la grammaire et les expressions amazighes avec des explications claires.",
    feat3T: "Traduction intelligente",
    feat3D:
      "Traduis entre le tamazight, le français, l'arabe et l'anglais instantanément.",
    feat4T: "Culture & patrimoine",
    feat4D:
      "Explore les proverbes, contes et traditions amazighes à travers des conversations enrichissantes.",
    feat5T: "Prononciation guidée",
    feat5D:
      "Comprends comment prononcer les mots et les phrases en tamazight correctement.",
    feat6T: "Multi-dialectes",
    feat6D:
      "Compatible avec le Tachelhit, le Tarifit, le Tamazight central et d'autres variantes.",
    featTitle: "Tout pour la langue amazighe",
    featSub:
      "Un assistant IA dédié à la langue, la culture et le patrimoine amazigh.",
    discover: "Découvrir",
    configure: "Apprendre",
    launch: "Pratiquer",
    optimize: "Maîtriser",
    discoverT: "Découvre la langue",
    discoverD:
      "Explore le tamazight latinisé avec un assistant qui comprend ta façon de parler.",
    configureT: "Apprends à ton rythme",
    configureD:
      "Vocabulaire, grammaire, expressions — progresse selon tes besoins.",
    launchT: "Pratique chaque jour",
    launchD:
      "Des conversations quotidiennes pour renforcer ta maîtrise de la langue.",
    optimizeT: "Deviens fluent",
    optimizeD:
      "Atteins un niveau naturel en amazigh grâce à une pratique régulière avec l'IA.",
    processTitle: "Comment ça marche",
    processSub:
      "De tes premiers mots à la fluidité, chaque étape est pensée pour toi.",
    testTitle: "Ce que disent nos utilisateurs",
    testSub:
      "Des Amazighs du monde entier qui apprennent et redécouvrent leur langue.",
    testText:
      "J'ai grandi sans vraiment parler tamazight. Grâce à ce chatbot, je peux enfin converser avec ma famille au village. L'IA comprend même le mélange darija-amazigh qu'on utilise tous les jours.",
    testName: "Mohamed Azelmad",
    testRole: "Utilisateur",
    ctaTitle: "Prêt à redécouvrir ta langue amazighe ?",
    ctaSub:
      "Commence à converser en tamazight dès aujourd'hui. Gratuit, simple et accessible partout.",
    newsletter: "Restez informé",
    emailPlaceholder: "Entrez votre email",
    getAccess: "Accès anticipé",
    rights: "Tous droits réservés",
    terms: "Conditions d'utilisation",
    privacy: "Confidentialité",
    users: "Locuteurs",
    uptime: "Disponibilité",
    support: "Support",
  },
  en: {
    features: "Features",
    process: "Process",
    testimonials: "Testimonials",
    login: "Login",
    startFree: "Start for free →",
    contact: "Contact Us",
    heroTitle: "Chat with AI in",
    heroRed: "Latinized Amazigh",
    heroSub:
      "Discover, learn and speak Tamazight with an intelligent AI assistant. Ask in latinized Amazigh — our AI understands and responds.",
    feat1T: "Natural conversation",
    feat1D: "Chat freely in latinized Amazigh. The AI adapts to how you write.",
    feat2T: "Language learning",
    feat2D:
      "Learn Amazigh vocabulary, grammar and expressions with clear, friendly explanations.",
    feat3T: "Smart translation",
    feat3D:
      "Translate between Tamazight, French, Arabic and English instantly.",
    feat4T: "Culture & heritage",
    feat4D:
      "Explore Amazigh proverbs, stories and traditions through enriching conversations.",
    feat5T: "Pronunciation guide",
    feat5D:
      "Understand how to correctly pronounce Tamazight words and sentences.",
    feat6T: "Multi-dialect support",
    feat6D:
      "Compatible with Tachelhit, Tarifit, Central Tamazight and other variants.",
    featTitle: "Everything for the Amazigh language",
    featSub:
      "An AI assistant dedicated to the Amazigh language, culture and heritage.",
    discover: "Discover",
    configure: "Learn",
    launch: "Practice",
    optimize: "Master",
    discoverT: "Discover the language",
    discoverD:
      "Explore latinized Tamazight with an assistant that understands how you speak.",
    configureT: "Learn at your pace",
    configureD:
      "Vocabulary, grammar, expressions — progress according to your needs.",
    launchT: "Practice every day",
    launchD: "Daily conversations to strengthen your command of the language.",
    optimizeT: "Become fluent",
    optimizeD:
      "Reach a natural level in Amazigh through regular AI-powered practice.",
    processTitle: "How it works",
    processSub:
      "From your first words to fluency, every step is designed for you.",
    testTitle: "What our users say",
    testSub:
      "Amazighs from around the world learning and rediscovering their language.",
    testText:
      "I grew up not really speaking Tamazight. Thanks to this chatbot, I can finally talk with my family in the village. The AI even understands the Darija-Amazigh mix we use every day.",
    testName: "Mohamed Azelmad",
    testRole: "User",
    ctaTitle: "Ready to rediscover your Amazigh language?",
    ctaSub:
      "Start chatting in Tamazight today. Free, simple and accessible everywhere.",
    newsletter: "Stay informed",
    emailPlaceholder: "Enter your email",
    getAccess: "Get Early Access",
    rights: "All rights reserved",
    terms: "Terms of service",
    privacy: "Privacy policy",
    users: "Speakers",
    uptime: "Uptime",
    support: "Support",
  },
  ar: {
    features: "الميزات",
    process: "العملية",
    testimonials: "شهادات",
    login: "تسجيل الدخول",
    startFree: "ابدأ مجانًا ←",
    contact: "اتصل بنا",
    heroTitle: "تحدث مع الذكاء الاصطناعي بـ",
    heroRed: "الأمازيغية اللاتينية",
    heroSub:
      "اكتشف وتعلم وتحدث بالتامازيغت مع مساعد ذكاء اصطناعي. اسأل بالأمازيغية اللاتينية — ذكاؤنا الاصطناعي يفهم ويجيب.",
    feat1T: "محادثة طبيعية",
    feat1D:
      "تحدث بحرية بالأمازيغية اللاتينية. الذكاء الاصطناعي يتكيف مع طريقة كتابتك.",
    feat2T: "تعلم اللغة",
    feat2D: "تعلم المفردات والقواعد والتعابير الأمازيغية بشروحات واضحة وسهلة.",
    feat3T: "ترجمة ذكية",
    feat3D: "ترجم بين التامازيغت والفرنسية والعربية والإنجليزية فورًا.",
    feat4T: "الثقافة والتراث",
    feat4D:
      "استكشف الأمثال والحكايات والتقاليد الأمازيغية من خلال محادثات ثرية.",
    feat5T: "دليل النطق",
    feat5D: "تعرف على كيفية نطق الكلمات والجمل الأمازيغية بشكل صحيح.",
    feat6T: "دعم متعدد اللهجات",
    feat6D: "متوافق مع تاشلحيت وتاريفيت وتامازيغت المركزية وغيرها.",
    featTitle: "كل شيء للغة الأمازيغية",
    featSub: "مساعد ذكاء اصطناعي مخصص للغة والثقافة والتراث الأمازيغي.",
    discover: "اكتشاف",
    configure: "تعلم",
    launch: "تدرب",
    optimize: "أتقن",
    discoverT: "اكتشف اللغة",
    discoverD: "استكشف التامازيغت اللاتينية مع مساعد يفهم طريقة كلامك.",
    configureT: "تعلم بوتيرتك",
    configureD: "مفردات وقواعد وتعابير — تقدم حسب احتياجاتك.",
    launchT: "تدرب كل يوم",
    launchD: "محادثات يومية لتعزيز إتقانك للغة.",
    optimizeT: "صِر طليقًا",
    optimizeD:
      "بلّغ مستوى طبيعيًا في الأمازيغية بممارسة منتظمة مع الذكاء الاصطناعي.",
    processTitle: "كيف يعمل",
    processSub: "من أولى كلماتك إلى الطلاقة، كل خطوة مصممة لك.",
    testTitle: "ما يقوله مستخدمونا",
    testSub: "أمازيغ من حول العالم يتعلمون ويعيدون اكتشاف لغتهم.",
    testText:
      "نشأت دون أن أتحدث التامازيغت حقًا. بفضل هذا الشات بوت، أستطيع أخيرًا التحدث مع عائلتي في القرية. الذكاء الاصطناعي يفهم حتى مزيج الدارجة والأمازيغية الذي نستخدمه يوميًا.",
    testName: "مخمد ازلماض",
    testRole: "مستخدم",
    ctaTitle: "هل أنت مستعد لإعادة اكتشاف لغتك الأمازيغية؟",
    ctaSub: "ابدأ التحدث بالتامازيغت اليوم. مجاني وبسيط ومتاح في كل مكان.",
    newsletter: "ابق على اطلاع",
    emailPlaceholder: "أدخل بريدك الإلكتروني",
    getAccess: "احصل على وصول مبكر",
    rights: "جميع الحقوق محفوظة",
    terms: "شروط الخدمة",
    privacy: "سياسة الخصوصية",
    users: "متحدث",
    uptime: "وقت التشغيل",
    support: "دعم",
  },
};
export default function Landing() {
  const nav = useNavigate();
  const { user } = useAuth();
  const { language, setLanguage } = useLanguage();
  const t = T[language];
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) =>
        entries.forEach(
          (e) => e.isIntersecting && e.target.classList.add("fadeInVisible"),
        ),
      { threshold: 0.1 },
    );
    document.querySelectorAll(".fadeIn").forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [language]);
  const features = [
    { t: t.feat1T, d: t.feat1D },
    { t: t.feat2T, d: t.feat2D },
    { t: t.feat3T, d: t.feat3D },
    { t: t.feat4T, d: t.feat4D },
    { t: t.feat5T, d: t.feat5D },
    { t: t.feat6T, d: t.feat6D },
  ];
  return (
    <div className="landingPage">
      <div className="landingBg1" />
      <div className="landingBg2" />
      {}
      <nav className="navMain">
        <div className="navLogo">
          <img src="/logo.png" alt="Logo" />
          <span>AWAL GPT</span>
        </div>
        <div className="navLinks">
          <a href="#features">{t.features}</a>
          <a href="#process">{t.process}</a>
          <a href="#testimonials">{t.testimonials}</a>
        </div>
        <div className="navActions">
          {["fr", "en", "ar"].map((l) => (
            <button
              key={l}
              onClick={() => setLanguage(l)}
              className={`btnLanguage ${language === l ? "btnLanguageActive" : ""}`}
            >
              {l.toUpperCase()}
            </button>
          ))}
          <button className="btnLogin" onClick={() => nav("/login")}>
            {t.login}
          </button>
        </div>
      </nav>
      {}
      <section className="heroSection">
        <div className="heroLeft">
          <h1 className="heroTitle">
            {t.heroTitle}
            <br />
            <span className="colorPrimary">{t.heroRed}</span>
          </h1>
          <p className="heroSub">{t.heroSub}</p>
          <div className="heroCta">
            <button
              className="btnHero"
              onClick={() => nav(user ? "/chat" : "/register")}
            >
              {t.startFree}
            </button>
            <button className="btnHeroOutline">{t.contact}</button>
          </div>
          <div className="heroStats">
            <div className="statItem">
              <span className="statNum">100+</span>
              <span className="statLbl">{t.users}</span>
            </div>
            <div className="statItem">
              <span className="statNum">99.9%</span>
              <span className="statLbl">{t.uptime}</span>
            </div>
            <div className="statItem">
              <span className="statNum">24/7</span>
              <span className="statLbl">{t.support}</span>
            </div>
          </div>
        </div>
        <div className="heroRight">
          <div className="chatCard">
            <div className="chatCardHeader">
              <img src="/logo.png" alt="Logo" width={24} />
              <span className="chatCardTitle">AWAL GPT</span>
              <span className="onlineDot" />
              <span style={{ fontSize: 12, color: "#6b7280" }}>Online</span>
            </div>
            <div className="chatMessages">
              <div className="userBubble">Amek ara iniɣ "merci" s tmaziɣt?</div>
              <div className="botBubble">
                Tzemreḍ ad tiniḍ "tanemmirt" — yella d awal amaziɣ aɣerfan. Deg
                tachelhit, nniḍen ttinin "barakallaufik".
              </div>
              <div className="userBubble">Yes! What does "azul" mean?</div>
              <div className="botBubble">
                "Azul" signifie "bonjour" en amazigh — c'est le mot de
                salutation le plus utilisé. Il vient du mot "azul" (le bleu du
                ciel).
              </div>
            </div>
            <div className="chatInputFake">
              <span>Ask me anything...</span>
              <div className="sendFake">→</div>
            </div>
          </div>
        </div>
      </section>
      {}
      <section className="section" id="features">
        <div className="fadeIn eyebrow">
          <div className="line" />
          {t.features}
        </div>
        <h2 className="fadeIn secTitle">{t.featTitle}</h2>
        <p className="fadeIn secSub">{t.featSub}</p>
        <div className="featGrid">
          {features.map((f, i) => (
            <div key={i} className="fadeIn featCard">
              <h3 className="featTitle">{f.t}</h3>
              <p className="featDesc">{f.d}</p>
            </div>
          ))}
        </div>
      </section>
      {}
      <section className="section sectionAlt" id="process">
        <div className="fadeIn eyebrow">
          <div className="line" />
          {t.process}
        </div>
        <h2 className="fadeIn secTitle">{t.processTitle}</h2>
        <p className="fadeIn secSub">{t.processSub}</p>
        <div className="processGrid">
          {[
            {
              n: "01",
              label: t.discover,
              title: t.discoverT,
              desc: t.discoverD,
            },
            {
              n: "02",
              label: t.configure,
              title: t.configureT,
              desc: t.configureD,
            },
            { n: "03", label: t.launch, title: t.launchT, desc: t.launchD },
            {
              n: "04",
              label: t.optimize,
              title: t.optimizeT,
              desc: t.optimizeD,
            },
          ].map((step, i) => (
            <div key={i} className="fadeIn processStep">
              <div className="processNum">{step.n}</div>
              <div className="processLabel">{step.label}</div>
              <div className="processTitle">{step.title}</div>
              <div className="processDesc">{step.desc}</div>
            </div>
          ))}
        </div>
      </section>
      {}
      <section className="section" id="testimonials">
        <div className="fadeIn eyebrow">
          <div className="line" />
          {t.testimonials}
        </div>
        <h2 className="fadeIn secTitle">{t.testTitle}</h2>
        <p className="fadeIn secSub">{t.testSub}</p>
        <div className="fadeIn testimonialCard">
          <div className="stars">★★★★★</div>
          <p className="testimonialText">"{t.testText}"</p>
          <div className="testimonialAuthor">
            <div className="authorAvatar">M</div>
            <div>
              <div className="authorName">{t.testName}</div>
              <div className="authorRole">{t.testRole}</div>
            </div>
          </div>
        </div>
      </section>
      {}
      <section className="ctaSection">
        <h2 className="fadeIn ctaTitle">{t.ctaTitle}</h2>
        <p className="fadeIn ctaSub">{t.ctaSub}</p>
        <button
          className="fadeIn btnHero"
          onClick={() => nav(user ? "/chat" : "/register")}
        >
          {t.startFree}
        </button>
      </section>
      {}
      <footer className="footer">
        <div className="footerGrid">
          <div>
            <div className="footerLogo">
              <img src="/logo.png" alt="Logo" width={28} />
              <span>AWAL GPT</span>
            </div>
            <p className="footerAbout">
              Plateforme IA pour la langue et la culture amazighe.
            </p>
          </div>
          <div className="footerCol">
            <h4>Product</h4>
            <a href="#features">{t.features}</a>
            <a href="#process">{t.process}</a>
          </div>
          <div className="footerCol">
            <h4>Company</h4>
            <a href="#testimonials">{t.testimonials}</a>
            <a href="#">Contact</a>
          </div>
          <div className="footerCol">
            <h4>{t.newsletter}</h4>
            <div className="newsletterInput">
              <input type="email" placeholder={t.emailPlaceholder} />
              <button>{t.getAccess}</button>
            </div>
          </div>
        </div>
        <div className="footerBottom">
          <span>© 2025 AWAL GPT · {t.rights}</span>
          <div className="footerLinks">
            <a href="#">{t.terms}</a>
            <a href="#">{t.privacy}</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
