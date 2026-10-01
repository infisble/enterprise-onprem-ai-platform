import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, errText, uuid } from "../api";
import { ROLE_LABEL, useAuth } from "../App";
import { Badge, CitationList, fmtMs, ProductCards, ToolCallsTable } from "../components";
import { Markdown } from "../markdown";
import type { ChatResponse, Scenario } from "../types";

interface Turn {
  id: string;
  user: string;
  role: string;
  message: string;
  response?: ChatResponse;
  error?: string;
  pending?: boolean;
}

const LATENCY_KEYS = ["total_ms", "ttft_ms", "retrieval_ms", "rerank_ms", "sql_ms", "llm_ms", "tool_ms"];

export function PendingApprovalCard({ p }: { p: Record<string, unknown> }) {
  const hash = String(p.payload_hash ?? "");
  return (
    <div className="pending-card">
      <div>
        <b>Change request #{String(p.request_id ?? "?")}</b> <Badge tone="accent">awaiting admin approval</Badge>
      </div>
      <div className="small">
        {p.sku ? <span className="mono">{String(p.sku)}</span> : null} {p.field ? <span className="mono">· {String(p.field)}</span> : null}{" "}
        · payload hash <span className="mono">{hash.slice(0, 12)}…</span> · base version {String(p.base_version ?? "?")}
        {p.requested_by ? ` · requested by ${String(p.requested_by)}` : ""}
      </div>
      <div className="small">
        Nothing was written to the product database. <Link to="/approvals">Open Approvals →</Link>
      </div>
    </div>
  );
}

export function ResponseDetails({ r }: { r: ChatResponse }) {
  const llm = r.llm ?? {};
  const total = Number(r.latency?.total_ms ?? 0);
  return (
    <div className="resp-details">
      {r.security_notices.map((n, i) => (
        <div key={i} className="alert alert-warn">
          ⚠ {n}
        </div>
      ))}
      {r.pending_approval && <PendingApprovalCard p={r.pending_approval} />}
      <ProductCards products={r.products ?? []} />
      <div className="meta-row">
        <Badge tone="route">route: {r.route}</Badge>
        {llm.router ? <span className="small">router: <b>{String(llm.router)}</b></span> : null}
        {llm.route_reason ? <span className="small muted">“{String(llm.route_reason)}”</span> : null}
        <span className="small muted mono">req {r.request_id.slice(0, 12)}</span>
      </div>
      <div className="latency-grid">
        {LATENCY_KEYS.filter((k) => r.latency?.[k] !== undefined).map((k) => (
          <div key={k} className="lat">
            <div className="lat-k">{k.replace("_ms", "")}</div>
            <div className="lat-v">{fmtMs(r.latency[k])}</div>
            {k !== "total_ms" && total > 0 && (
              <div className="lat-bar">
                <div style={{ width: `${Math.min(100, (Number(r.latency[k]) / total) * 100)}%` }} />
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="small">
        LLM: <b>{String(llm.provider || "—")}</b> / <span className="mono">{String(llm.model || "—")}</span> · tokens in{" "}
        {String(llm.input_tokens ?? 0)} / out {String(llm.output_tokens ?? 0)}
      </div>
      {r.tool_calls.length > 0 && (
        <details open>
          <summary>Tool calls ({r.tool_calls.length})</summary>
          <ToolCallsTable calls={r.tool_calls} />
        </details>
      )}
      {r.citations.length > 0 && (
        <details open>
          <summary>Citations ({r.citations.length})</summary>
          <CitationList citations={r.citations} />
        </details>
      )}
    </div>
  );
}

export default function ChatPage() {
  const { user, switchUser } = useAuth();
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [convId, setConvId] = useState(() => uuid());
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api<Scenario[]>("/api/scenarios").then(setScenarios).catch(() => setScenarios([]));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const send = async (message: string) => {
    if (!message.trim() || busy || !user) return;
    const id = uuid();
    setTurns((t) => [...t, { id, user: user.username, role: user.role, message, pending: true }]);
    setInput("");
    setBusy(true);
    try {
      const r = await api<ChatResponse>("/api/chat", { body: { message, conversation_id: convId } });
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, response: r, pending: false } : x)));
    } catch (e) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: errText(e), pending: false } : x)));
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div className="chat-page">
      <div className="page-head">
        <h1>AI Chat</h1>
        <div className="small muted">
          conversation <span className="mono">{convId.slice(0, 8)}</span>{" "}
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              setTurns([]);
              setConvId(uuid());
            }}
          >
            New conversation
          </button>
        </div>
      </div>

      <div className="scenarios">
        {scenarios.map((s) => {
          const mismatch = user && s.user !== user.username;
          return (
            <div key={s.id} className={`chip${mismatch ? " chip-other" : ""}`} title={s.demonstrates}>
              <button className="chip-main" onClick={() => setInput(s.message)}>
                <b>
                  {s.id}. {s.title}
                </b>
                <span className="chip-user">as {s.user}</span>
              </button>
              {mismatch && (
                <button
                  className="chip-switch"
                  title={`Switch to ${s.user} and fill the message`}
                  onClick={async () => {
                    try {
                      await switchUser(s.user);
                      setInput(s.message);
                    } catch {
                      /* surfaced in top bar */
                    }
                  }}
                >
                  switch
                </button>
              )}
            </div>
          );
        })}
      </div>

      <div className="chat-log">
        {turns.length === 0 && (
          <div className="empty muted">
            Pick a scenario above or ask a question. Try the same question as different users to see RBAC in action.
          </div>
        )}
        {turns.map((t) => (
          <div key={t.id} className="turn">
            <div className="msg msg-user">
              <div className="msg-who">
                {t.user} <span className={`role role-${t.role}`}>{ROLE_LABEL[t.role] ?? t.role}</span>
              </div>
              <div>{t.message}</div>
            </div>
            <div className="msg msg-bot">
              {t.pending && <div className="muted">Thinking…</div>}
              {t.error && <div className="alert alert-error">{t.error}</div>}
              {t.response && (
                <>
                  <Markdown text={t.response.answer} />
                  <ResponseDetails r={t.response} />
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form className="chat-input" onSubmit={onSubmit}>
        <textarea
          value={input}
          placeholder="Ask about products, stock, policies, or request content…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          rows={2}
        />
        <button className="btn" disabled={busy || !input.trim()}>
          {busy ? "Sending…" : "Send"}
        </button>
      </form>
    </div>
  );
}
