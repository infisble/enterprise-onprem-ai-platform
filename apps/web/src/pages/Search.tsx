import { Fragment, useEffect, useState, type ChangeEvent, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, errText, qs } from "../api";
import { Badge, ErrorBox, fmtMs, fmtNum, Json } from "../components";
import type { Facets, SearchResponse } from "../types";

interface FilterState {
  price_min: string;
  price_max: string;
  color: string;
  category: string;
  material: string;
  size: string;
  in_stock: boolean;
}

const EMPTY: FilterState = { price_min: "", price_max: "", color: "", category: "", material: "", size: "", in_stock: false };

function StageCard({ name, data }: { name: string; data: Record<string, unknown> }) {
  const { ms, ...rest } = data;
  return (
    <div className="stage">
      <div className="stage-head">
        <b>{name}</b>
        {ms !== undefined && <span className="mono small">{fmtMs(ms)}</span>}
      </div>
      <dl className="kv tight">
        {Object.entries(rest).map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd className="mono small">{typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function SearchPage() {
  const [q, setQ] = useState("black metal bottle 750 ml under 15 EUR");
  const [mode, setMode] = useState("hybrid");
  const [limit, setLimit] = useState(12);
  const [f, setF] = useState<FilterState>(EMPTY);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [facets, setFacets] = useState<Facets | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  useEffect(() => {
    api<Facets>("/api/search/facets").then(setFacets).catch(() => setFacets(null));
  }, []);

  const run = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const params = qs({
        q,
        mode,
        limit,
        price_min: f.price_min,
        price_max: f.price_max,
        color: f.color,
        category: f.category,
        material: f.material,
        size: f.size,
        in_stock: f.in_stock ? true : undefined,
      });
      setRes(await api<SearchResponse>(`/api/search/products${params}`));
      setOpen({});
    } catch (ex) {
      setErr(errText(ex));
    } finally {
      setBusy(false);
    }
  };

  const set = (k: keyof FilterState) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setF({ ...f, [k]: k === "in_stock" ? (e.target as HTMLInputElement).checked : e.target.value });

  const activeFilters = res ? Object.entries(res.query.filters).filter(([, v]) => v !== null && v !== "") : [];

  return (
    <div className="with-side">
      <div className="grow">
        <h1>Product Search</h1>
        <form className="card search-form" onSubmit={run}>
          <div className="row">
            <input className="grow" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Natural-language query, SKU…" />
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="hybrid">hybrid</option>
              <option value="lexical">lexical (BM25)</option>
              <option value="vector">vector</option>
            </select>
            <input type="number" min={1} max={50} value={limit} onChange={(e) => setLimit(Number(e.target.value))} style={{ width: 70 }} title="limit" />
            <button className="btn" disabled={busy}>
              {busy ? "Searching…" : "Search"}
            </button>
          </div>
          <div className="row filters">
            <label>price min<input type="number" step="0.01" value={f.price_min} onChange={set("price_min")} /></label>
            <label>price max<input type="number" step="0.01" value={f.price_max} onChange={set("price_max")} /></label>
            <label>colour
              <select value={f.color} onChange={set("color")}>
                <option value="">any</option>
                {Object.keys(facets?.color ?? {}).map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label>category
              <select value={f.category} onChange={set("category")}>
                <option value="">any</option>
                {Object.keys(facets?.category ?? {}).map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label>material<input value={f.material} onChange={set("material")} list="materials" /></label>
            <datalist id="materials">
              {Object.keys(facets?.material ?? {}).map((m) => <option key={m} value={m} />)}
            </datalist>
            <label>size<input value={f.size} onChange={set("size")} style={{ width: 60 }} /></label>
            <label className="check"><input type="checkbox" checked={f.in_stock} onChange={set("in_stock")} /> in stock</label>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setF(EMPTY)}>clear</button>
          </div>
        </form>
        <ErrorBox error={err} />

        {res && (
          <>
            <div className="card">
              <div className="meta-row">
                <span>
                  semantic text: <span className="mono">{res.query.semantic_text || "—"}</span>
                </span>
                <span className="small muted">lang {res.query.detected_language}</span>
                {res.query.exact_code && <Badge tone="accent">exact code {res.query.exact_code}</Badge>}
                <span className="small">
                  {res.total_candidates} candidates · <b>{fmtMs(res.latency_ms)}</b>
                </span>
              </div>
              <div className="meta-row">
                <span className="small muted">extracted filters:</span>
                {activeFilters.length ? (
                  activeFilters.map(([k, v]) => (
                    <Badge key={k} tone="neutral">
                      {k}={String(v)}
                    </Badge>
                  ))
                ) : (
                  <span className="small muted">none</span>
                )}
              </div>
              {res.relaxed_filters.length > 0 && (
                <div className="alert alert-warn">Relaxed filters (no exact matches): {res.relaxed_filters.join(", ")}</div>
              )}
              {res.query.notes.length > 0 && <div className="small muted">{res.query.notes.join(" · ")}</div>}
              <div className="small muted">
                backends: lexical <b>{res.lexical_backend}</b> · vector <b>{res.vector_backend}</b> · reranker <b>{res.reranker}</b>
              </div>
            </div>

            <h3>Pipeline stages</h3>
            <div className="stages">
              {Object.entries(res.stages).map(([k, v]) => (
                <StageCard key={k} name={k} data={v ?? {}} />
              ))}
            </div>

            <h3>Results ({res.results.length})</h3>
            <table className="table">
              <thead>
                <tr>
                  <th>SKU</th>
                  <th>Name</th>
                  <th className="num">Price</th>
                  <th className="num">Stock</th>
                  <th>Colour</th>
                  <th>Material</th>
                  <th className="num">Capacity</th>
                  <th className="num">Score</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {res.results.map((p) => (
                  <Fragment key={p.sku}>
                    <tr>
                      <td className="mono">
                        <Link to={`/products/${encodeURIComponent(p.sku)}`}>{p.sku}</Link>
                      </td>
                      <td>{p.name}</td>
                      <td className="num">{p.price_eur.toFixed(2)}</td>
                      <td className="num" title={p.stock_available === null ? "not permitted for your role" : ""}>
                        {p.stock_available === null ? "—" : p.stock_available.toLocaleString()}
                      </td>
                      <td>{p.color}</td>
                      <td>{p.material}</td>
                      <td className="num">{p.capacity_ml ? `${p.capacity_ml} ml` : ""}</td>
                      <td className="num mono">{p.final_score.toFixed(4)}</td>
                      <td>
                        <button className="btn btn-ghost btn-sm" onClick={() => setOpen({ ...open, [p.sku]: !open[p.sku] })}>
                          {open[p.sku] ? "hide" : "why?"}
                        </button>
                      </td>
                    </tr>
                    {open[p.sku] && (
                      <tr className="explain-row">
                        <td colSpan={9}>
                          <div className="explain">
                            <span>BM25 rank <b>{p.explain.bm25_rank ?? "—"}</b> ({fmtNum(p.explain.bm25_score, 3)})</span>
                            <span>vector rank <b>{p.explain.vector_rank ?? "—"}</b> ({fmtNum(p.explain.vector_score, 4)})</span>
                            <span>RRF <b>{fmtNum(p.explain.rrf_score, 5)}</b></span>
                            <span>rerank <b>{fmtNum(p.explain.rerank_score, 4)}</b></span>
                            <span>business adj. <b>{p.explain.business_adjustment >= 0 ? "+" : ""}{fmtNum(p.explain.business_adjustment, 3)}</b></span>
                            <span>
                              matched:{" "}
                              {p.explain.matched_attributes.length
                                ? p.explain.matched_attributes.map((a) => <Badge key={a}>{a}</Badge>)
                                : "—"}
                            </span>
                          </div>
                          <details>
                            <summary className="small">raw</summary>
                            <Json value={p} />
                          </details>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <aside className="side-panel card">
        <h3>Facets</h3>
        {!facets && <div className="muted small">unavailable</div>}
        {facets &&
          Object.entries(facets).map(([name, values]) => (
            <div key={name} className="facet">
              <div className="facet-name">{name}</div>
              {Object.entries(values).map(([v, n]) => (
                <div key={v} className="facet-row">
                  <span>{v}</span>
                  <span className="mono muted">{n}</span>
                </div>
              ))}
            </div>
          ))}
      </aside>
    </div>
  );
}
