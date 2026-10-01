import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, errText, qs } from "../api";
import { useAuth } from "../App";
import { Badge, ErrorBox, fmtTime, StatusBadge } from "../components";
import type { UpdateRequest } from "../types";

const TABS = ["pending", "published", "rejected", "conflict", "all"];

export function can(perms: string[] | undefined, p: string): boolean {
  return !!perms?.includes(p);
}

function RequestCard({ r, canApprove, onDone }: { r: UpdateRequest; canApprove: boolean; onDone: (msg: string) => void }) {
  const { user } = useAuth();
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const v = r.validation ?? {};
  const pending = r.status === "pending" || r.status === "approved";
  const ownRequest = user?.username === r.requested_by;

  const decide = async (action: "approve" | "reject") => {
    setBusy(true);
    setErr(null);
    setOk(null);
    try {
      const res = await api<UpdateRequest>(`/api/approvals/${r.id}/${action}`, { body: { note } });
      setOk(`Request #${r.id} → ${res.status}`);
      onDone(`Request #${r.id} (${r.sku} ${r.field}) → ${res.status}`);
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card approval">
      <div className="row between">
        <div className="meta-row">
          <b>#{r.id}</b>
          <Link className="mono" to={`/products/${encodeURIComponent(r.sku)}`}>
            {r.sku}
          </Link>
          <span className="mono">{r.field}</span>
          <StatusBadge status={r.status} />
          <Badge tone={r.source === "agent" ? "orange" : "neutral"}>source: {r.source}</Badge>
        </div>
        <div className="small muted">
          requested by <b>{r.requested_by}</b> · {fmtTime(r.requested_at)}
        </div>
      </div>
      {r.rationale && <div className="small">Rationale: {r.rationale}</div>}
      <div className="small muted">
        payload hash <span className="mono">{r.payload_hash.slice(0, 16)}…</span> · base version {r.base_version} · request{" "}
        <span className="mono">{r.request_id.slice(0, 12)}</span>
      </div>

      <div className="grid-validation">
        <div>
          <div className="small">
            <b>Validation</b>{" "}
            {v.valid === undefined ? null : v.valid ? <Badge tone="ok">valid</Badge> : <Badge tone="bad">invalid</Badge>}
            {v.length !== undefined && <span className="muted"> · {String(v.length)} chars</span>}
          </div>
          <ul className="checks">
            {Object.entries(v.checks ?? {}).map(([k, pass]) => (
              <li key={k} className={pass ? "text-ok" : "text-bad"}>
                {pass ? "✓" : "✗"} <span className="mono small">{k}</span>
              </li>
            ))}
          </ul>
          {(v.errors ?? []).map((e, i) => (
            <div key={i} className="small text-bad">
              error: {e}
            </div>
          ))}
          {(v.warnings ?? []).map((w, i) => (
            <div key={i} className="small text-warn">
              warning: {w}
            </div>
          ))}
        </div>
        <div className="diff">
          <div>
            <div className="diff-h">OLD</div>
            <div className="diff-old">{r.old_value || <span className="muted">(empty)</span>}</div>
          </div>
          <div>
            <div className="diff-h">NEW</div>
            <div className="diff-new">{r.new_value}</div>
          </div>
        </div>
      </div>

      {r.decided_by && (
        <div className="small muted">
          decided by <b>{r.decided_by}</b> at {fmtTime(r.decided_at)}
          {r.decision_note && <> — “{r.decision_note}”</>}
          {r.published_at && <> · published {fmtTime(r.published_at)}</>}
        </div>
      )}

      {pending && (
        <div className="decision">
          <input placeholder="Decision note (optional)" value={note} onChange={(e) => setNote(e.target.value)} disabled={!canApprove || busy} />
          <button className="btn" disabled={!canApprove || busy} onClick={() => decide("approve")}>
            Approve & Publish
          </button>
          <button className="btn btn-danger" disabled={!canApprove || busy} onClick={() => decide("reject")}>
            Reject
          </button>
          {!canApprove && <span className="small muted">Only administrators (updates:approve) can decide.</span>}
          {canApprove && ownRequest && (
            <span className="small text-warn">You requested this change — separation of duties will block approval.</span>
          )}
        </div>
      )}
      <ErrorBox error={err} />
      {ok && <div className="alert alert-ok">{ok}</div>}
    </div>
  );
}

export default function ApprovalsPage() {
  const { user } = useAuth();
  const [tab, setTab] = useState("pending");
  const [rows, setRows] = useState<UpdateRequest[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  const canApprove = can(user?.permissions, "updates:approve");

  const load = useCallback(() => {
    setLoading(true);
    api<UpdateRequest[]>(`/api/approvals${qs({ status: tab === "all" ? undefined : tab })}`)
      .then((r) => {
        setRows(r);
        setErr(null);
      })
      .catch((e) => {
        setRows([]);
        setErr(errText(e));
      })
      .finally(() => setLoading(false));
  }, [tab]);

  useEffect(load, [load]);

  return (
    <div>
      <div className="page-head">
        <h1>Approvals</h1>
        <button className="btn btn-ghost btn-sm" onClick={load}>
          Refresh
        </button>
      </div>
      {!canApprove && (
        <div className="alert alert-info">
          You are signed in as <b>{user?.role}</b>. You can see your own change requests but cannot approve or reject — switch
          to <span className="mono">adam.admin</span> to decide.
        </div>
      )}
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>
      <ErrorBox error={err} />
      {flash && <div className="alert alert-ok">{flash}</div>}
      {loading && <div className="muted small">Loading…</div>}
      {!loading && !err && rows.length === 0 && <div className="muted">No {tab === "all" ? "" : tab} requests.</div>}
      {rows.map((r) => (
        <RequestCard key={r.id} r={r} canApprove={canApprove} onDone={(m) => { setFlash(m); load(); }} />
      ))}
    </div>
  );
}
