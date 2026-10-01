import { useState, type FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { errText } from "../api";
import { ROLE_LABEL, useAuth } from "../App";
import { ErrorBox } from "../components";

const ROLE_HINT: Record<string, string> = {
  employee: "Search products & public/internal documents. No stock, pricing or approvals.",
  sales: "Adds live stock, price tiers and confidential sales documents.",
  catalog_manager: "Generates and proposes product content; cannot publish own changes.",
  admin: "Approves & publishes changes, runs batch jobs, analytics SQL, all traces.",
};

export default function LoginPage() {
  const { demoUsers, switchUser } = useAuth();
  const navigate = useNavigate();
  const loc = useLocation();
  const from = (loc.state as { from?: string } | null)?.from ?? "/chat";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("demo");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const doLogin = async (u: string, p: string) => {
    setBusy(true);
    setErr(null);
    try {
      await switchUser(u, p);
      navigate(from === "/login" ? "/chat" : from, { replace: true });
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (username) doLogin(username, password);
  };

  return (
    <div className="login-page">
      <div className="login-card card">
        <div className="brand-name big">NovaPromo</div>
        <div className="muted">Enterprise on-prem AI platform — demo sign-in</div>
        <h3>Demo users (password “demo”)</h3>
        <div className="demo-users">
          {demoUsers.map((u) => (
            <button key={u.username} className="demo-user" disabled={busy} onClick={() => doLogin(u.username, "demo")}>
              <div>
                <b>{u.full_name}</b> <span className={`role role-${u.role}`}>{ROLE_LABEL[u.role] ?? u.role}</span>
              </div>
              <div className="mono small">{u.username}</div>
              <div className="small muted">{ROLE_HINT[u.role] ?? ""}</div>
            </button>
          ))}
          {!demoUsers.length && <div className="muted">Demo users unavailable — is the API running?</div>}
        </div>
        <form onSubmit={onSubmit} className="login-form">
          <input placeholder="username" value={username} onChange={(e) => setUsername(e.target.value)} />
          <input placeholder="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          <button className="btn" disabled={busy || !username}>
            Sign in
          </button>
        </form>
        <ErrorBox error={err} />
      </div>
    </div>
  );
}
