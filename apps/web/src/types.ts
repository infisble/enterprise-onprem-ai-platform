export interface User {
  username: string;
  role: string;
  full_name: string;
  permissions?: string[];
}

export interface DemoUser {
  username: string;
  full_name: string;
  role: string;
}

export interface Scenario {
  id: number;
  title: string;
  user: string;
  message: string;
  demonstrates: string;
}

export interface Citation {
  ref: number;
  source: string;
  doc_type: string;
  locator: Record<string, unknown>;
  snippet: string;
  score: number;
}

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  outcome: string;
  side_effect: string;
  duration_ms: number;
  result_preview?: string;
  error?: string;
}

export interface RankExplanation {
  bm25_rank: number | null;
  bm25_score: number | null;
  vector_rank: number | null;
  vector_score: number | null;
  rrf_score: number;
  rerank_score: number | null;
  business_adjustment: number;
  matched_attributes: string[];
}

export interface ProductHit {
  product_id: number;
  sku: string;
  name: string;
  name_en: string;
  category: string;
  color: string;
  material: string;
  capacity_ml: number | null;
  sizes: string[];
  price_eur: number;
  stock_available: number | null;
  supplier: string;
  brand: string;
  lead_time_days: number;
  tags: string[];
  final_score: number;
  explain: RankExplanation;
}

export interface ChatResponse {
  request_id: string;
  answer: string;
  route: string;
  citations: Citation[];
  tool_calls: ToolCall[];
  products: ProductHit[];
  pending_approval: Record<string, unknown> | null;
  security_notices: string[];
  latency: Record<string, number>;
  llm: Record<string, unknown>;
}

export interface SearchFilters {
  [k: string]: string | number | boolean | null | undefined;
}

export interface SearchResponse {
  query: {
    original: string;
    semantic_text: string;
    filters: SearchFilters;
    detected_language: string;
    exact_code: string | null;
    notes: string[];
  };
  results: ProductHit[];
  total_candidates: number;
  stages: Record<string, Record<string, unknown>>;
  relaxed_filters: string[];
  lexical_backend: string;
  vector_backend: string;
  reranker: string;
  latency_ms: number;
}

export type Facets = Record<string, Record<string, number>>;

export interface ProductDetail {
  product_id: number;
  sku: string;
  supplier_sku: string;
  supplier: string;
  name_sk: string;
  name_en: string;
  name_cs: string;
  description_sk: string;
  description_en: string;
  description_cs: string;
  category: string;
  subcategory: string;
  color: string;
  material: string;
  capacity_ml: number | null;
  sizes: string[];
  price_eur: number;
  brand: string;
  country_of_origin: string;
  lead_time_days: number;
  tags: string[];
  status: string;
  content_source: string;
  needs_review: boolean;
  version: number;
  stock_available: number | null;
  stock: { sku: string; total_available: number; by_warehouse: Record<string, number>; lead_time_days: number; as_of: string } | null;
  pricing: { sku: string; list_price_eur: number; tiers: { min_qty: number; unit_price_eur: number }[]; currency: string } | null;
  update_requests: { id: number; field: string; status: string; requested_by: string; requested_at: string }[];
}

export interface TraceRow {
  request_id: string;
  ts: string;
  user: string;
  role: string;
  kind: string;
  query: string;
  route: string;
  llm_provider: string;
  model: string;
  total_ms: number;
  input_tokens: number;
  output_tokens: number;
  tools: string[];
  error: string;
}

export interface Span {
  name: string;
  kind: string;
  attrs: Record<string, unknown>;
  start_ms: number;
  duration_ms: number;
}

export interface TraceDetail extends Omit<TraceRow, "tools" | "kind"> {
  kind?: string;
  spans: Span[];
  tool_calls: ToolCall[];
}

export interface UpdateRequest {
  id: number;
  sku: string;
  field: string;
  old_value: string;
  new_value: string;
  payload_hash: string;
  base_version: number;
  status: string;
  source: string;
  validation: {
    valid?: boolean;
    errors?: string[];
    warnings?: string[];
    checks?: Record<string, boolean>;
    length?: number;
    [k: string]: unknown;
  };
  rationale: string;
  request_id: string;
  requested_by: string;
  requested_at: string;
  decided_by: string | null;
  decided_at: string | null;
  decision_note: string;
  published_at: string | null;
}

export interface BatchJob {
  id: number;
  job_type: string;
  status: string;
  params: Record<string, unknown>;
  created_by: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  total: number;
  processed: number;
  succeeded: number;
  failed: number;
  skipped: number;
  checkpoint_product_id: number;
  stats: Record<string, number | string | null>;
  error: string;
}

export interface BatchItem {
  product_id: number;
  status: string;
  attempts: number;
  actions: string[];
  error: string;
}

export interface BatchJobDetail {
  job: BatchJob;
  item_counts: Record<string, number>;
  recent_items: BatchItem[];
  failed_items: BatchItem[];
}

export interface SystemInfo {
  llm: { provider: string; model: string; routing: string };
  embeddings: string;
  reranker: string;
  lexical_backend: string;
  vector_backend: string;
  cache_backend: string;
  batch_mode: string;
  env: string;
}

export interface Readiness {
  status: string;
  checks: Record<string, { ok: boolean; backend?: string; error?: string; [k: string]: unknown }>;
}

export interface MetricsSummary {
  window: {
    requests: number;
    requests_per_min: number;
    latency_ms?: { p50: number; p95: number; mean?: number };
    llm_latency_ms?: { p50: number; p95: number };
    retrieval_latency_ms?: { p50: number; p95: number };
    output_tokens?: number;
    tokens_per_sec?: number;
    error_rate?: number;
    routes?: Record<string, number>;
  };
  batch: { queue_size: number; jobs: number; items_processed: number; success_rate: number | null; running: number };
  audit_outcomes: Record<string, number>;
  routes_all_time: Record<string, number>;
  gpu: Record<string, unknown>;
  gpu_sources?: Record<string, string>;
}
