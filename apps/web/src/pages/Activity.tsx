import { useCallback, useEffect, useState } from "react";
import { api, errText } from "../api";
import { Badge, ErrorBox, fmtMs, fmtTime, Json, StatusBadge, ToolCallsTable, useInterval } from "../components";
import type { TraceDetail, TraceRow } from "../types";

const KINDS = ["retrieval", "rerank", "sql", "llm", "tool", "internal"];

function Timeline({ t }: { t: TraceDetail }) {
  const end = Math.max(t.total_ms || 0, ...t.spans.map((s) => s.start_ms + s.duration_ms), 1);
  return (
    <div className="timeline">
      <div className="legend">
        {KINDS.map((k) => (
          <span key={k} className="legend-item">
            <span className={`swatch span-${k}`} /> {k}
          </span>
        ))}
      </div>
      {t.spans.map((s, i) => (
        <div key={i} className="tl-row" title={JSON.stringify(s.attrs)}>
          <div className="tl-label">
            <span className="mono">{s.name}</span>
          </div>
          <div className="tl-track">
            <div
              className={`tl-bar span-${KINDS.includes(s.kind) ? s.kind : "internal"}`}
              style={{
                left: `${(s.start_ms / end) * 100}%`,
                width: `${Math.max(0.4, (s.duration_ms / end) * 100)}%`,
              }}
            />
          </div>
          <div className="tl-ms mono small">{fmtMs(s.duration_ms)}</div>
        </div>
      ))}
      <div className="tl-axis small muted">
        <span>0 ms</span>
        <span>{fmtMs(end)}</span>
      </div>
      <details>
        <summary className="small">span attributes</summary>
        <table className="table compact">
          <tbody>
            {t.spans.map((s, i) => (
              <tr key={i}>
                <td className="mono">{s.name}</td>
                <td>{s.kind}</td>
                <td className="num">@{fmtMs(s.start_ms)}</td>
                <td>
                  <Json value={s.attrs} compact />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

export default function ActivityPage() {
  const [rows, setRows] = useState<TraceRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [auto, setAuto] = useState(false);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [detailErr, setDetailErr] = useState<string | null>(null);

  const load = useCallback(() => {
    api<TraceRow[]>("/api/traces?limit=100")
      .then((r) => {
        setRows(r);
        setErr(null);
      })
      .catch((e) => setErr(errText(e)));
  }, []);

  useEffect(load, [load]);
  useInterval(load, auto ? 3000 : null);

  useEffect(() => {
    if (!sel) return;
    setDetail(null);
    setDetailErr(null);
    api<TraceDetail>(`/api/traces/${sel}`).then(setDetail).catch((e) => setDetailErr(errText(e)));
  }, [sel]);

  return (
    <div>
      <div className="page-head">
        <h1>Agent Activity</h1>
        <div className="row">
          <label className="check">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> auto-refresh (3 s)
          </label>
          <button className="btn btn-ghost btn-sm" onClick={load}>
            Refresh
          </button>
        </div>
      </div>
      <ErrorBox error={err} />
      <div className="small muted">Non-admin users see only their own requests.</div>
      <div className="split">
        <div className="split-left">
          <table className="table compact clickable">
            <thead>
              <tr>
                <th>Time</th>
                <th>User</th>
                <th>Kind</th>
                <th>Route</th>
                <th>Query</th>
                <th className="num">Total</th>
                <th className="num">Tokens</th>
                <th>Tools</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.request_id} className={sel === r.request_id ? "selected" : ""} onClick={() => setSel(r.request_id)}>
                  <td className="nowrap small">{fmtTime(r.ts)}</td>
                  <td>
                    {r.user}
                    <div className="small muted">{r.role}</div>
                  </td>
                  <td>{r.kind}</td>
                  <td>
                    <Badge tone="route">{r.route || "—"}</Badge>
                  </td>
                  <td className="query-cell">
                    {r.query}
                    {r.error && <div className="small text-bad">{r.error}</div>}
                  </td>
                  <td className="num">{fmtMs(r.total_ms)}</td>
                  <td className="num small">
                    {r.input_tokens}/{r.output_tokens}
                  </td>
                  <td className="small">
                    {r.tools.map((t, i) => {
                      const [name, outcome] = t.split(":");
                      return (
                        <div key={i} className={`tool-${outcome}`}>
                          <span className="mono">{name}</span> <StatusBadge status={outcome ?? ""} />
                        </div>
                      );
                    })}
                  </td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={8} className="muted">
                    No traces yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="split-right">
          {!sel && <div className="card muted">Select a request to see its span timeline.</div>}
          <ErrorBox error={detailErr} />
          {sel && !detail && !detailErr && <div className="card muted">Loading…</div>}
          {detail && (
            <div className="card">
              <h3>
                Request <span className="mono">{detail.request_id.slice(0, 12)}</span>
              </h3>
              <div className="small">{detail.query}</div>
              <div className="meta-row">
                <Badge tone="route">{detail.route}</Badge>
                <span className="small">
                  {detail.user} ({detail.role})
                </span>
                <span className="small">total {fmtMs(detail.total_ms)}</span>
                <span className="small">
                  {detail.llm_provider || "no llm"} {detail.model && <span className="mono">/ {detail.model}</span>} · tokens{" "}
                  {detail.input_tokens}/{detail.output_tokens}
                </span>
              </div>
              {detail.error && <div className="alert alert-error">{detail.error}</div>}
              <Timeline t={detail} />
              {detail.tool_calls.length > 0 && (
                <>
                  <h4>Tool calls</h4>
                  <ToolCallsTable calls={detail.tool_calls} />
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
