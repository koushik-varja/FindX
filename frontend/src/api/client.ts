import type {
  EvaluationPayload,
  IndexStatus,
  SearchLabPayload,
  SearchPayload,
} from "../types/search";

export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`FindX API request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function textSearch(query: string): Promise<SearchPayload> {
  return parse(
    await fetch(`${API}/api/search/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 12, mode: "reranked", debug: true }),
    }),
  );
}

export async function searchLab(query: string): Promise<SearchLabPayload> {
  return parse(
    await fetch(`${API}/api/search/lab`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 6 }),
    }),
  );
}

export async function imageSearch(file: File, text = "", k = 12): Promise<SearchPayload> {
  const form = new FormData();
  form.append("file", file);
  form.append("k", String(k));
  if (text) {
    form.append("text", text);
  }
  const endpoint = text ? "/api/search/multimodal" : "/api/search/image";
  return parse(await fetch(`${API}${endpoint}`, { method: "POST", body: form }));
}

export async function latestEvaluation(): Promise<EvaluationPayload> {
  return parse(await fetch(`${API}/api/evaluation/latest`));
}

export async function indexStatus(): Promise<IndexStatus> {
  return parse(await fetch(`${API}/api/index/status`));
}

export function imageUrl(path: string): string {
  return `${API}/demo-images/${path.split("/").pop()}`;
}
