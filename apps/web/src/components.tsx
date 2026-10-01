import { useEffect, useRef, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Citation, ProductHit, ToolCall } from "./types";

export function fmtMs(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 2 : 1)} ms`;
}

export function fmtNum(v: unknown, digits = 2): string {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(digits);
}

export function fmtPct(v: unknown): string {
  if (v === null || v === undefined) return "—";
  return `${(Number(v) * 100).toFixed(1)} %`;
}

/** Backend timestamps are naive UTC ISO strings. */
export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + "Z");
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function Badge({ children, tone = "neutral", title }: { children: ReactNode; tone?: string; title?: string }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

const STATUS_TONE: Record<string, string> = {
  success: "ok", succeeded: "ok", published: "ok", completed: "ok", ready: "ok", approved: "ok",
  denied: "bad", failed: "bad", completed_with_errors: "warn", rejected: "bad", invalid_args: "warn",
  error: "warn", conflict: "warn", degraded: "warn", paused: "warn",
  pending: "accent", running: "accent", queued: "accent",
};

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={STATUS_TONE[status] ?? "neutral"}>{status}</Badge>;
}

export function ErrorBox({ error }: { error: string | null | undefined }) {
  if (!error) return null;
  return <div className="alert alert-error">{error}</div>;
}

export function Json({ value, compact }: { value: unknown; compact?: boolean }) {
  return <pre className={`json${compact ? " compact" : ""}`}>{JSON.stringify(value, null, compact ? 0 : 2)}</pre>;
}

export function Kv({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

export function useInterval(fn: () => void, ms: number | null) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (ms === null) return;
    const id = window.setInterval(() => ref.current(), ms);
    return () => window.clearInterval(id);
  }, [ms]);
}

export function ToolCallsTable({ calls }: { calls: ToolCall[] }) {
  if (!calls.length) return null;
  return (
    <table className="table compact">
      <thead>
        <tr>
          <th>Tool</th>
          <th>Side effect</th>
          <th>Outcome</th>
          <th className="num">Duration</th>
          <th>Args / result</th>
        </tr>
      </thead>
      <tbody>
        {calls.map((c, i) => (
          <tr key={i}>
            <td className="mono">{c.tool}</td>
            <td>
              <Badge tone={c.side_effect === "write" ? "orange" : "neutral"}>{c.side_effect}</Badge>
            </td>
            <td>
              <StatusBadge status={c.outcome} />
            </td>
            <td className="num">{fmtMs(c.duration_ms)}</td>
            <td>
              <Json value={c.args} compact />
              {c.error && <div className="small text-bad">{c.error}</div>}
              {c.result_preview && (
                <details>
                  <summary className="small">result preview</summary>
                  <pre className="json">{c.result_preview}</pre>
                </details>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function locatorLabel(loc: Record<string, unknown>): string {
  const parts: string[] = [];
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  for (const key of ["sheet", "row", "page", "section", "path"]) {
    const v = loc[key];
    if (v === null || v === undefined || v === "") continue;
    parts.push(key === "sheet" || key === "page" || key === "row" ? `${cap(key)} ${v}` : `${cap(key)}: ${v}`);
  }
  for (const [k, v] of Object.entries(loc)) {
    if (!["sheet", "row", "page", "section", "path"].includes(k) && v !== null && v !== "") parts.push(`${k}: ${v}`);
  }
  return parts.join(" · ");
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <ol className="citations">
      {citations.map((c) => (
        <li key={c.ref}>
          <div>
            <span className="cite-ref">[{c.ref}]</span> <strong>{c.source}</strong>{" "}
            <span className="muted">{locatorLabel(c.locator)}</span>{" "}
            <span className="muted small">score {fmtNum(c.score, 3)}</span>
          </div>
          <div className="snippet">{c.snippet}</div>
        </li>
      ))}
    </ol>
  );
}

export function ProductCards({ products }: { products: ProductHit[] }) {
  if (!products.length) return null;
  return (
    <div className="product-cards">
      {products.map((p) => (
        <Link key={p.sku} to={`/products/${encodeURIComponent(p.sku)}`} className="product-card">
          <div className="mono sku">{p.sku}</div>
          <div className="pc-name">{p.name}</div>
          <div className="pc-meta">
            {p.color} · {p.material}
            {p.capacity_ml ? ` · ${p.capacity_ml} ml` : ""}
          </div>
          <div className="pc-foot">
            <strong>{p.price_eur.toFixed(2)} EUR</strong>
            <span className={p.stock_available === null ? "muted" : ""}>
              {p.stock_available === null ? "stock —" : `${p.stock_available.toLocaleString()} pcs`}
            </span>
          </div>
        </Link>
      ))}
    </div>
  );
}

export function Progress({ value, max, parts }: { value?: number; max: number; parts?: { v: number; tone: string }[] }) {
  const pct = (v: number) => `${max > 0 ? Math.min(100, (v / max) * 100) : 0}%`;
  return (
    <div className="progress">
      {parts
        ? parts.map((p, i) => <div key={i} className={`bar bar-${p.tone}`} style={{ width: pct(p.v) }} />)
        : <div className="bar bar-accent" style={{ width: pct(value ?? 0) }} />}
    </div>
  );
}
