import { useCallback, useEffect, useState } from "react";
import { api, errText } from "../api";
import { Badge, ErrorBox, fmtMs, fmtNum, fmtPct, Json, useInterval } from "../components";
import type { MetricsSummary, Readiness, SystemInfo } from "../types";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function Bars({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  if (!entries.length) return <div className="muted small">no data</div>;
  return (
    <div className="bars">
      {entries.map(([k, v]) => (
        <div key={k} className="bars-row">
          <span className="bars-label">{k || "(none)"}</span>
          <div className="bars-track">
            <div className="bars-fill" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="bars-v mono">{v}</span>
        </div>
      ))}
    </div>
  );
}

export default function MetricsPage() {
  const [m, setM] = useState<MetricsSummary | null>(null);
  const [ready, setReady] = useState<Readiness | null>(null);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [auto, setAuto] = useState(true);

  const load = useCallback(() => {
    Promise.all([
      api<MetricsSummary>("/api/metrics/summary"),
      api<Readiness>("/health/ready", { auth: false }),
      api<SystemInfo>("/api/system/info"),
    ])
      .then(([a, b, c]) => {
        setM(a);
        setReady(b);
        setInfo(c);
        setErr(null);
      })
      .catch((e) => setErr(errText(e)));
  }, []);

  useEffect(load, [load]);
  useInterval(load, auto ? 5000 : null);

  const w = m?.window;
  const gpuNote = m?.gpu && typeof m.gpu.note === "string" ? m.gpu.note : null;

  return (
    <div>
      <div className="page-head">
        <h1>System Metrics</h1>
        <div className="row">
          <label className="check">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> auto-refresh (5 s)
          </label>
          <a href="/metrics" target="_blank" rel="noreferrer">
            raw Prometheus /metrics
          </a>
        </div>
      </div>
      <ErrorBox error={err} />
      <div className="small muted">
        Rolling window since API start. With the monitoring compose profile enabled, Grafana dashboards are on{" "}
        <a href="http://localhost:3001" target="_blank" rel="noreferrer">
          :3001
        </a>
        .
      </div>

      {w && (
        <div className="stats">
          <Stat label="Requests / min" value={fmtNum(w.requests_per_min)} sub={`${w.requests} in window`} />
          <Stat label="Latency p50 / p95" value={`${fmtMs(w.latency_ms?.p50)} / ${fmtMs(w.latency_ms?.p95)}`} sub={w.latency_ms?.mean !== undefined ? `mean ${fmtMs(w.latency_ms.mean)}` : undefined} />
          <Stat label="LLM p50 / p95" value={`${fmtMs(w.llm_latency_ms?.p50)} / ${fmtMs(w.llm_latency_ms?.p95)}`} />
          <Stat label="Retrieval p50 / p95" value={`${fmtMs(w.retrieval_latency_ms?.p50)} / ${fmtMs(w.retrieval_latency_ms?.p95)}`} />
          <Stat label="Tokens / sec" value={fmtNum(w.tokens_per_sec, 1)} sub={w.output_tokens !== undefined ? `${w.output_tokens} output tokens` : undefined} />
          <Stat label="Error rate" value={w.error_rate === undefined ? "—" : fmtPct(w.error_rate)} />
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h3>Route distribution (window)</h3>
          <Bars data={w?.routes ?? {}} />
          <h3>Routes (all time)</h3>
          <Bars data={m?.routes_all_time ?? {}} />
        </div>
        <div className="card">
          <h3>Batch</h3>
          {m && (
            <div className="stats compact">
              <Stat label="Queue size" value={fmtNum(m.batch.queue_size)} />
              <Stat label="Running jobs" value={fmtNum(m.batch.running)} />
              <Stat label="Items processed" value={fmtNum(m.batch.items_processed)} sub={`last ${m.batch.jobs} jobs`} />
              <Stat label="Success rate" value={fmtPct(m.batch.success_rate)} />
            </div>
          )}
          <h3>Audit outcomes</h3>
          <Bars data={m?.audit_outcomes ?? {}} />
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>
            Dependency health{" "}
            {ready && <Badge tone={ready.status === "ready" ? "ok" : "warn"}>{ready.status}</Badge>}
          </h3>
          <table className="table compact">
            <tbody>
              {ready &&
                Object.entries(ready.checks).map(([k, c]) => {
                  const { ok, backend, error, ...rest } = c;
                  return (
                    <tr key={k}>
                      <td>
                        <b>{k}</b>
                      </td>
                      <td>{ok ? <Badge tone="ok">ok</Badge> : <Badge tone="bad">down</Badge>}</td>
                      <td className="mono">{backend ?? (rest.provider as string) ?? ""}</td>
                      <td className="small">
                        {error ? <span className="text-bad">{error}</span> : Object.entries(rest).map(([rk, rv]) => `${rk}: ${String(rv)}`).join(" · ")}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
          {info && (
            <>
              <h3>Active providers</h3>
              <Json value={info} />
            </>
          )}
        </div>
        <div className="card">
          <h3>GPU</h3>
          {gpuNote ? <div className="alert alert-info">{gpuNote}</div> : m && <Json value={m.gpu} />}
          {m && typeof m.gpu.kv_cache_usage === "number" && (
            <div className="stats compact">
              <Stat label="vLLM KV cache" value={fmtPct(m.gpu.kv_cache_usage as number)} />
              <Stat label="Requests running" value={String(m.gpu.requests_running ?? "—")} />
              <Stat label="Requests waiting" value={String(m.gpu.requests_waiting ?? "—")} />
            </div>
          )}
          {m?.gpu_sources && (
            <ul className="muted small">
              {Object.entries(m.gpu_sources).map(([k, v]) => (
                <li key={k}>
                  <b>{k.replace(/_/g, " ")}</b>: {v}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
