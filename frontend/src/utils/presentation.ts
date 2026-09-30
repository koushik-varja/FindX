import type {
  EvaluationPayload,
  IndexStatus,
  SearchResult,
} from "../types/search";

export const labModes = ["bm25", "dense", "hybrid", "reranked"] as const;

export type EvaluationRow = {
  mode: string;
  recall: number;
  mrr: number;
  ndcg: number;
  p50: number;
  p95: number;
};

export function correctionLabel(original?: string, corrected?: string): string {
  if (!corrected || !original || original === corrected) return original || corrected || "";
  return `Showing results for “${corrected}” · original “${original}”`;
}

export function filterChips(filters: Record<string, unknown> = {}): string[] {
  return Object.entries(filters).map(([key, value]) => {
    const label = key.replaceAll("_", " ");
    if (key.startsWith("price_") && typeof value === "number") return `${label}: ₹${value}`;
    return `${label}: ${String(value)}`;
  });
}

export function resultReason(result: SearchResult): string {
  return result.debug?.reason || "ranked result";
}

export function evaluationRows(data: EvaluationPayload | null): EvaluationRow[] {
  if (!data?.modes) return [];
  return Object.entries(data.modes).map(([mode, metrics]) => ({
    mode,
    recall: metrics.recall_at_k,
    mrr: metrics.mrr,
    ndcg: metrics.ndcg_at_k,
    p50: metrics.latency_ms_p50,
    p95: metrics.latency_ms_p95,
  }));
}

export function runtimeBadge(status: IndexStatus | null): string {
  return status?.runtime_label || (status?.mode === "full" ? "FULL ML" : "LIGHTWEIGHT");
}
