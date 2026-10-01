import { createContext, Fragment, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, clearAuth, errText, getStoredUser, getToken, login } from "./api";
import type { DemoUser, SystemInfo, User } from "./types";
import LoginPage from "./pages/Login";
import ChatPage from "./pages/Chat";
import SearchPage from "./pages/Search";
import ProductPage from "./pages/Product";
import ActivityPage from "./pages/Activity";
import ApprovalsPage from "./pages/Approvals";
import BatchPage from "./pages/Batch";
import MetricsPage from "./pages/Metrics";

interface AuthCtx {
  user: User | null;
  demoUsers: DemoUser[];
  switchUser: (username: string, password?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthCtx>({
  user: null,
  demoUsers: [],
  switchUser: async () => {},
  logout: () => {},
});

export const useAuth = () => useContext(AuthContext);

export const ROLE_LABEL: Record<string, string> = {
  employee: "Employee",
  sales: "Sales",
  catalog_manager: "Catalog manager",
  admin: "Admin",
};

const NAV = [
  { to: "/chat", label: "AI Chat" },
  { to: "/search", label: "Product Search" },
  { to: "/activity", label: "Agent Activity" },
  { to: "/approvals", label: "Approvals" },
  { to: "/batch", label: "Batch Jobs" },
  { to: "/system", label: "System Metrics" },
];

function TopBar() {
  const { user, demoUsers, switchUser, logout } = useAuth();
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<SystemInfo>("/api/system/info").then(setInfo).catch(() => setInfo(null));
  }, [user?.username]);

  const onSwitch = async (username: string) => {
    if (!username || username === user?.username) return;
    setBusy(true);
    setErr(null);
    try {
      await switchUser(username);
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <header className="topbar">
      <div className="providers">
        {info ? (
          <>
            <span>
              LLM: <b>{info.llm.provider}</b> <span className="muted">({info.llm.model})</span>
            </span>
            <span>
              Lexical: <b>{info.lexical_backend}</b>
            </span>
            <span>
              Vector: <b>{info.vector_backend}</b>
            </span>
            <span>
              Rerank: <b>{info.reranker}</b>
            </span>
            <span>
              Router: <b>{info.llm.routing}</b>
            </span>
          </>
        ) : (
          <span className="muted">providers unavailable</span>
        )}
      </div>
      <div className="userbox">
        {err && <span className="text-bad small">{err}</span>}
        {user && (
          <span className="current-user">
            <b>{user.full_name}</b> <span className={`role role-${user.role}`}>{ROLE_LABEL[user.role] ?? user.role}</span>
          </span>
        )}
        <label className="switcher">
          <span className="small muted">Switch user</span>
          <select value={user?.username ?? ""} disabled={busy} onChange={(e) => onSwitch(e.target.value)}>
            {demoUsers.map((u) => (
              <option key={u.username} value={u.username}>
                {u.username} ({ROLE_LABEL[u.role] ?? u.role})
              </option>
            ))}
          </select>
        </label>
        <button className="btn btn-ghost" onClick={logout}>
          Log out
        </button>
      </div>
    </header>
  );
}

function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-name">NovaPromo</div>
          <div className="brand-sub">On-prem AI Platform</div>
        </div>
        <nav>
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} className={({ isActive }) => (isActive ? "active" : "")}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot small">
          <a href="/metrics" target="_blank" rel="noreferrer">
            Prometheus /metrics
          </a>
        </div>
      </aside>
      <div className="main">
        <TopBar />
        <main className="content">{children}</main>
      </div>
    </div>
  );
}

/** Pages are remounted on user switch (fresh data for the new role) unless keep=true (chat keeps its history). */
function RequireAuth({ children, keep }: { children: ReactNode; keep?: boolean }) {
  const { user } = useAuth();
  const loc = useLocation();
  if (!user || !getToken()) return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  return (
    <Shell>
      <Fragment key={keep ? "keep" : user.username}>{children}</Fragment>
    </Shell>
  );
}

export default function App() {
  const [user, setUser] = useState<User | null>(() => (getToken() ? getStoredUser() : null));
  const [demoUsers, setDemoUsers] = useState<DemoUser[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    api<DemoUser[]>("/api/auth/demo-users", { auth: false })
      .then(setDemoUsers)
      .catch(() => setDemoUsers([]));
  }, []);

  useEffect(() => {
    const onLogout = () => {
      setUser(null);
      navigate("/login");
    };
    window.addEventListener("auth:logout", onLogout);
    return () => window.removeEventListener("auth:logout", onLogout);
  }, [navigate]);

  const switchUser = useCallback(async (username: string, password = "demo") => {
    const u = await login(username, password);
    setUser(u);
  }, []);

  const logout = useCallback(() => {
    clearAuth();
    setUser(null);
    navigate("/login");
  }, [navigate]);

  return (
    <AuthContext.Provider value={{ user, demoUsers, switchUser, logout }}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<RequireAuth keep><ChatPage /></RequireAuth>} />
        <Route path="/search" element={<RequireAuth><SearchPage /></RequireAuth>} />
        <Route path="/products/:sku" element={<RequireAuth><ProductPage /></RequireAuth>} />
        <Route path="/activity" element={<RequireAuth><ActivityPage /></RequireAuth>} />
        <Route path="/approvals" element={<RequireAuth><ApprovalsPage /></RequireAuth>} />
        <Route path="/batch" element={<RequireAuth><BatchPage /></RequireAuth>} />
        <Route path="/system" element={<RequireAuth><MetricsPage /></RequireAuth>} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </AuthContext.Provider>
  );
}
