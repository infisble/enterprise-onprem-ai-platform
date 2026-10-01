import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, errText } from "../api";
import { useAuth } from "../App";
import { Badge, ErrorBox, fmtNum, fmtPct, fmtTime, Json, Progress, StatusBadge, useInterval } from "../components";
import type { BatchJob, BatchJobDetail } from "../types";
import { can } from "./Approvals";

const ACTIVE = ["running", "queued"];

function JobDetail({ id, tick }: { id: number; tick: number }) {
  const [d, setD] = useState<BatchJobDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<BatchJobDetail>(`/api/batch/jobs/${id}`)
      .then((x) => {
        setD(x);
        setErr(null);
      })
      .catch((e) => setErr(errText(e)));
  }, [id, tick]);
  if (err) return <ErrorBox error={err} />;
  if (!d) return <div className="muted">Loading…</div>;
  return (
    <div className="card">
      <h3>Job #{d.job.id} detail</h3>
      <div className="meta-row">
        {Object.entries(d.item_counts).map(([k, v]) => (
          <span key={k}>
            <StatusBadge status={k} /> <b>{v}</b>
          </span>
        ))}
      </div>
      <div className="small">
        params: <Json value={d.job.params} compact />
      </div>
      {d.job.error && <div className="alert alert-error">{d.job.error}</div>}
      <h4>Recent items</h4>
      <table className="table compact">
        <thead>
          <tr>
            <th>Product</th>
            <th>Status</th>
            <th className="num">Attempts</th>
            <th>Actions</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody>
          {d.recent_items.map((i, n) => (
            <tr key={n}>
              <td className="mono">{i.product_id}</td>
              <td>
                <StatusBadge status={i.status} />
              </td>
              <td className="num">{i.attempts}</td>
              <td>
                {i.actions.map((a, k) => (
                  <Badge key={k}>{a}</Badge>
                ))}
              </td>
              <td className="small text-bad">{i.error}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>Failed items ({d.failed_items.length})</h4>
      {d.failed_items.length ? (
        <table className="table compact">
          <tbody>
            {d.failed_items.map((i, n) => (
              <tr key={n}>
                <td className="mono">{i.product_id}</td>
                <td className="num">{i.attempts} attempts</td>
                <td className="small text-bad">{i.error}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="muted small">none</div>
      )}
    </div>
  );
}

export default function BatchPage() {
  const { user } = useAuth();
  const canRun = can(user?.permissions, "batch:run");
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [actionErr, setActionErr] = useState<string | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const [limit, setLimit] = useState("50");
  const [onlyMissing, setOnlyMissing] = useState(true);
  const [failRate, setFailRate] = useState(0.1);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api<BatchJob[]>("/api/batch/jobs")
      .then((j) => {
        setJobs(j);
        setErr(null);
        setTick((t) => t + 1);
      })
      .catch((e) => setErr(errText(e)));
  }, []);

  useEffect(load, [load]);
  const anyActive = jobs.some((j) => ACTIVE.includes(j.status));
  useInterval(load, anyActive ? 2000 : null);

  const start = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setActionErr(null);
    try {
      const j = await api<BatchJob>("/api/batch/jobs", {
        body: { job_type: "nightly_enrichment", limit: limit ? Number(limit) : null, only_missing: onlyMissing, fail_rate: failRate },
      });
      setSel(j.id);
      load();
    } catch (ex) {
      setActionErr(errText(ex));
    } finally {
      setBusy(false);
    }
  };

  const act = async (id: number, action: "pause" | "resume") => {
    setActionErr(null);
    try {
      await api(`/api/batch/jobs/${id}/${action}`, { method: "POST" });
      load();
    } catch (ex) {
      setActionErr(errText(ex));
    }
  };

  return (
    <div>
      <div className="page-head">
        <h1>Batch Jobs</h1>
        <div className="small muted">{anyActive ? "auto-refreshing every 2 s" : ""}</div>
      </div>
      <ErrorBox error={err} />
      <form className="card row" onSubmit={start}>
        <b>Start nightly enrichment</b>
        <label>
          limit <input type="number" min={1} value={limit} onChange={(e) => setLimit(e.target.value)} style={{ width: 80 }} />
        </label>
        <label className="check">
          <input type="checkbox" checked={onlyMissing} onChange={(e) => setOnlyMissing(e.target.checked)} /> only missing
        </label>
        <label>
          fault injection {failRate.toFixed(2)}
          <input type="range" min={0} max={0.5} step={0.05} value={failRate} onChange={(e) => setFailRate(Number(e.target.value))} />
        </label>
        <button className="btn" disabled={!canRun || busy}>
          Start job
        </button>
        {!canRun && <span className="small muted">requires batch:run (admin)</span>}
      </form>
      <ErrorBox error={actionErr} />
      <table className="table clickable">
        <thead>
          <tr>
            <th>#</th>
            <th>Type</th>
            <th>Status</th>
            <th style={{ width: "22%" }}>Progress</th>
            <th className="num">OK / fail / skip</th>
            <th className="num">items/s</th>
            <th className="num">Tokens</th>
            <th className="num">Success</th>
            <th className="num">Run</th>
            <th>Created</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.id} className={sel === j.id ? "selected" : ""} onClick={() => setSel(j.id)}>
              <td>{j.id}</td>
              <td className="mono small">{j.job_type}</td>
              <td>
                <StatusBadge status={j.status} />
              </td>
              <td>
                <Progress
                  max={Math.max(j.total, j.processed, 1)}
                  parts={[
                    { v: j.succeeded, tone: "ok" },
                    { v: j.skipped, tone: "neutral" },
                    { v: j.failed, tone: "bad" },
                  ]}
                />
                <div className="small muted">
                  {j.processed} / {j.total} processed
                </div>
              </td>
              <td className="num">
                <span className="text-ok">{j.succeeded}</span> / <span className="text-bad">{j.failed}</span> / {j.skipped}
              </td>
              <td className="num">{fmtNum(j.stats.items_per_sec)}</td>
              <td className="num">{fmtNum(j.stats.tokens)}</td>
              <td className="num">{fmtPct(j.stats.success_rate)}</td>
              <td className="num">{j.stats.run_seconds !== undefined ? `${fmtNum(j.stats.run_seconds)} s` : "—"}</td>
              <td className="small">
                {fmtTime(j.created_at)}
                <div className="muted">{j.created_by}</div>
              </td>
              <td onClick={(e) => e.stopPropagation()}>
                {ACTIVE.includes(j.status) && (
                  <button className="btn btn-ghost btn-sm" disabled={!canRun} onClick={() => act(j.id, "pause")}>
                    Pause
                  </button>
                )}
                {["paused", "failed", "completed_with_errors"].includes(j.status) && (
                  <button className="btn btn-ghost btn-sm" disabled={!canRun} onClick={() => act(j.id, "resume")}>
                    Resume
                  </button>
                )}
              </td>
            </tr>
          ))}
          {!jobs.length && !err && (
            <tr>
              <td colSpan={11} className="muted">
                No jobs yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      {sel !== null && <JobDetail id={sel} tick={tick} />}
    </div>
  );
}
