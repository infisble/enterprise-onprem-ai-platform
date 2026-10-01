import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, errText } from "../api";
import { Badge, ErrorBox, fmtTime, Kv, StatusBadge } from "../components";
import { Markdown } from "../markdown";
import type { ChatResponse, ProductDetail } from "../types";
import { ResponseDetails } from "./Chat";

export default function ProductPage() {
  const { sku = "" } = useParams();
  const [p, setP] = useState<ProductDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [gen, setGen] = useState<ChatResponse | null>(null);
  const [genErr, setGenErr] = useState<string | null>(null);
  const [genBusy, setGenBusy] = useState(false);

  const load = useCallback(() => {
    setErr(null);
    api<ProductDetail>(`/api/products/${encodeURIComponent(sku)}`)
      .then(setP)
      .catch((e) => {
        setP(null);
        setErr(errText(e));
      });
  }, [sku]);

  useEffect(load, [load]);

  const generate = async () => {
    setGenBusy(true);
    setGenErr(null);
    setGen(null);
    try {
      const r = await api<ChatResponse>("/api/chat", { body: { message: `Generate a new Slovak description for ${sku}.` } });
      setGen(r);
      load();
    } catch (e) {
      setGenErr(errText(e));
    } finally {
      setGenBusy(false);
    }
  };

  if (err) return <ErrorBox error={err} />;
  if (!p) return <div className="muted">Loading…</div>;

  return (
    <div>
      <div className="page-head">
        <h1>
          <span className="mono">{p.sku}</span> {p.name_en}
        </h1>
        <div className="meta-row">
          <Badge tone={p.content_source === "ai" ? "orange" : "neutral"}>content: {p.content_source}</Badge>
          {p.needs_review && <Badge tone="warn">needs review</Badge>}
          <Badge tone="accent">version {p.version}</Badge>
          <StatusBadge status={p.status} />
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Attributes</h3>
          <Kv
            items={[
              ["Supplier", `${p.supplier} (${p.supplier_sku})`],
              ["Brand", p.brand],
              ["Category", `${p.category}${p.subcategory ? " / " + p.subcategory : ""}`],
              ["Colour", p.color],
              ["Material", p.material],
              ["Capacity", p.capacity_ml ? `${p.capacity_ml} ml` : "—"],
              ["Sizes", p.sizes.length ? p.sizes.join(", ") : "—"],
              ["List price", `${p.price_eur.toFixed(2)} EUR`],
              ["Country of origin", p.country_of_origin],
              ["Lead time", `${p.lead_time_days} days`],
              ["Tags", p.tags.map((t) => <Badge key={t}>{t}</Badge>)],
              ["Stock available", p.stock_available === null ? <span className="muted">not permitted for your role</span> : p.stock_available.toLocaleString()],
            ]}
          />
        </div>
        <div>
          <div className="card">
            <h3>Stock by warehouse</h3>
            {p.stock ? (
              <>
                <table className="table compact">
                  <tbody>
                    {Object.entries(p.stock.by_warehouse).map(([w, n]) => (
                      <tr key={w}>
                        <td className="mono">{w}</td>
                        <td className="num">{n.toLocaleString()}</td>
                      </tr>
                    ))}
                    <tr>
                      <td><b>Total</b></td>
                      <td className="num"><b>{p.stock.total_available.toLocaleString()}</b></td>
                    </tr>
                  </tbody>
                </table>
                <div className="small muted">as of {fmtTime(p.stock.as_of)}</div>
              </>
            ) : (
              <div className="muted">Not permitted for your role.</div>
            )}
          </div>
          <div className="card">
            <h3>Pricing tiers</h3>
            {p.pricing ? (
              <table className="table compact">
                <thead>
                  <tr>
                    <th className="num">Min qty</th>
                    <th className="num">Unit price ({p.pricing.currency})</th>
                  </tr>
                </thead>
                <tbody>
                  {p.pricing.tiers.map((t) => (
                    <tr key={t.min_qty}>
                      <td className="num">{t.min_qty.toLocaleString()}</td>
                      <td className="num">{t.unit_price_eur.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="muted">Not permitted for your role.</div>
            )}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Names & descriptions</h3>
        <div className="grid-3">
          {(["sk", "en", "cs"] as const).map((l) => (
            <div key={l} className="lang-col">
              <div className="lang-tag">{l.toUpperCase()}</div>
              <div><b>{p[`name_${l}`] || <span className="muted">—</span>}</b></div>
              <p>{p[`description_${l}`] || <span className="muted">(missing)</span>}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="row between">
          <h3>AI content generation</h3>
          <button className="btn btn-accent2" onClick={generate} disabled={genBusy}>
            {genBusy ? "Generating…" : "Generate Slovak description with AI"}
          </button>
        </div>
        <div className="small muted">Creates a change request — nothing is published without admin approval.</div>
        <ErrorBox error={genErr} />
        {gen && (
          <div className="msg msg-bot">
            <Markdown text={gen.answer} />
            <ResponseDetails r={gen} />
          </div>
        )}
      </div>

      <div className="card">
        <h3>Update requests</h3>
        {p.update_requests.length ? (
          <table className="table compact">
            <thead>
              <tr>
                <th>#</th>
                <th>Field</th>
                <th>Status</th>
                <th>Requested by</th>
                <th>Requested at</th>
              </tr>
            </thead>
            <tbody>
              {p.update_requests.map((r) => (
                <tr key={r.id}>
                  <td>
                    <Link to="/approvals">#{r.id}</Link>
                  </td>
                  <td className="mono">{r.field}</td>
                  <td>
                    <StatusBadge status={r.status} />
                  </td>
                  <td>{r.requested_by}</td>
                  <td>{fmtTime(r.requested_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="muted">No change requests.</div>
        )}
      </div>
    </div>
  );
}
