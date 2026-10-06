export type Page = "search" | "lab" | "evaluation" | "index";
export type RuntimeMode = "lightweight" | "full";
export type SearchMode = "bm25" | "dense" | "hybrid" | "reranked";

export type SearchResultDebug = {
  lexical_score?: number;
  semantic_score?: number;
  fusion_score?: number;
  rerank_score?: number | null;
  matched_attributes?: string[];
  reason?: string;
  visual_similarity?: number;
  clip_text_image_similarity?: number;
  attribute_score?: number;
  bm25_rank?: number | null;
  dense_rank?: number | null;
  hybrid_rank?: number | null;
  final_rank?: number | null;
  rerank_components?: Record<string, number>;
};

export type SearchTimings = Record<string, number>;

export type SearchResult = {
  id: string;
  title: string;
  description: string;
  category: string;
  brand: string;
  price: number;
  colour: string;
  image_path: string;
  score: number;
  rank: number;
  debug?: SearchResultDebug;
};

export type SearchPayload = {
  mode: string;
  runtime_mode?: RuntimeMode;
  original_query?: string;
  corrected_query?: string;
  parsed_attributes?: Record<string, unknown>;
  latency_ms?: number;
  timings_ms?: SearchTimings;
  fusion_weights?: Record<string, number>;
  results: SearchResult[];
};

export type SearchLabPayload = Record<SearchMode, SearchPayload>;

export type EvaluationModeMetrics = {
  precision_at_k?: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  latency_ms_p99?: number | null;
};

export type ImageEvaluation = {
  metric: string;
  value: number;
  queries: number;
  label_origin: string;
};

export type EvaluationPayload = {
  label_origin?: string;
  queries: number;
  k: number;
  runtime_mode: RuntimeMode;
  modes: Record<string, EvaluationModeMetrics>;
  generated_at?: string;
  image_retrieval?: ImageEvaluation;
  artifact_mode_matches_runtime?: boolean;
};

export type IndexStatus = Record<string, unknown> & {
  runtime_label: "FULL ML" | "LIGHTWEIGHT";
  mode: RuntimeMode;
  index_version: string;
  indexed_products: number;
  lexical_index: string;
  dense_model: string;
  visual_model: string;
  text_vector_index: string;
  image_vector_index: string;
  text_embedding_dimension: number;
  image_embedding_dimension: number;
  loaded_from_persisted_index: boolean;
  last_rebuild_at: string;
  startup_seconds: number;
  numeric_vector_bytes: number;
};
