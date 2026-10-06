import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import * as client from "../api/client";
import type { EvaluationPayload, IndexStatus, SearchLabPayload, SearchPayload, SearchResult } from "../types/search";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    textSearch: vi.fn(),
    searchLab: vi.fn(),
    imageSearch: vi.fn(),
    latestEvaluation: vi.fn(),
    indexStatus: vi.fn(),
  };
});

const product: SearchResult = {
  id: "p001",
  title: "Samsung Wireless Earbuds",
  description: "Compact wireless earbuds for everyday listening.",
  category: "earbuds",
  brand: "Samsung",
  price: 2499,
  colour: "black",
  image_path: "data/demo/images/p001.png",
  score: 0.934,
  rank: 1,
  debug: {
    lexical_score: 0.82,
    semantic_score: 0.91,
    fusion_score: 0.03,
    rerank_score: 0.934,
    matched_attributes: ["brand=Samsung"],
    reason: "hybrid relevance + brand",
    bm25_rank: 2,
    dense_rank: 1,
    hybrid_rank: 1,
    final_rank: 1,
  },
};

function searchPayload(overrides: Partial<SearchPayload> = {}): SearchPayload {
  return {
    mode: "reranked",
    runtime_mode: "full",
    original_query: "samsoong wirless earbuds",
    corrected_query: "Samsung wireless earbuds",
    parsed_attributes: { brand: "Samsung", category: "earbuds" },
    latency_ms: 12.4,
    timings_ms: { preprocessing: 0.2, bm25: 0.4, dense_embedding: 8.1, dense_retrieval: 0.3, fusion: 0.1, ranking: 0.5, total: 9.6 },
    results: [product],
    ...overrides,
  };
}

function labPayload(): SearchLabPayload {
  return {
    bm25: searchPayload({ mode: "bm25" }),
    dense: searchPayload({ mode: "dense" }),
    hybrid: searchPayload({ mode: "hybrid" }),
    reranked: searchPayload({ mode: "reranked" }),
  };
}

const evaluation: EvaluationPayload = {
  queries: 60,
  k: 10,
  runtime_mode: "full",
  modes: {
    bm25: { recall_at_k: 0.8, mrr: 0.7, ndcg_at_k: 0.75, latency_ms_p50: 4.2, latency_ms_p95: 8.8 },
    reranked: { recall_at_k: 0.95, mrr: 0.9, ndcg_at_k: 0.92, latency_ms_p50: 8.4, latency_ms_p95: 14.1 },
  },
};

const fullStatus: IndexStatus = {
  runtime_label: "FULL ML",
  mode: "full",
  index_version: "full-test",
  indexed_products: 82,
  lexical_index: "custom-bm25-v1",
  dense_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  visual_model: "open_clip:ViT-B-32/laion2b_s34b_b79k",
  text_vector_index: "faiss-hnsw-ip-v1",
  image_vector_index: "faiss-hnsw-ip-v1",
  text_embedding_dimension: 384,
  image_embedding_dimension: 512,
  loaded_from_persisted_index: true,
  last_rebuild_at: "2026-09-08T00:00:00Z",
  startup_seconds: 1.2,
  numeric_vector_bytes: 1024,
};

beforeEach(() => {
  vi.mocked(client.textSearch).mockResolvedValue(searchPayload());
  vi.mocked(client.searchLab).mockResolvedValue(labPayload());
  vi.mocked(client.imageSearch).mockResolvedValue(searchPayload({ mode: "image", original_query: undefined, corrected_query: undefined }));
  vi.mocked(client.latestEvaluation).mockResolvedValue(evaluation);
  vi.mocked(client.indexStatus).mockResolvedValue(fullStatus);
});

describe("FindX frontend behaviour", () => {
  it("submits typo-tolerant text search without debug by default", async () => {
    const user = userEvent.setup();
    render(<App />);
    const query = screen.getByRole("textbox", { name: "search query" });
    await user.clear(query);
    await user.type(query, "samsoong wirless earbuds");
    const searchBox = query.parentElement;
    expect(searchBox).not.toBeNull();
    await user.click(within(searchBox as HTMLElement).getByRole("button", { name: "Search" }));
    expect(client.textSearch).toHaveBeenCalledWith("samsoong wirless earbuds", false);
    expect(await screen.findByText(/Showing results for “Samsung wireless earbuds”/)).toBeInTheDocument();
  });

  it("requests debug metadata when developer details are enabled", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("checkbox"));

const query = screen.getByRole("textbox", { name: "search query" });
const searchBox = query.parentElement;
expect(searchBox).not.toBeNull();

await user.click(
  within(searchBox as HTMLElement).getByRole("button", { name: "Search" }),
);
    expect(client.textSearch).toHaveBeenCalledWith("black running shoes under 3000", true);
    expect(await screen.findByText("dense_embedding")).toBeInTheDocument();
  });

  it("renders all four live Search Lab modes and the rank trace", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Search Lab" }));
    await user.click(screen.getByRole("button", { name: "Compare" }));
    expect(client.searchLab).toHaveBeenCalledWith("samsoong wirless earbuds under 3000", true);
    expect(await screen.findByRole("heading", { name: "BM25" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Dense ANN" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Hybrid RRF" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Reranked" })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "ranking trace" })).toBeInTheDocument();
  });

  it("uploads an image and calls image search", async () => {
    const user = userEvent.setup();
    render(<App />);
    const file = new File(["image-bytes"], "shoe.png", { type: "image/png" });
    await user.upload(screen.getByLabelText("product image"), file);
    await user.click(screen.getByRole("button", { name: "Find similar" }));
    await waitFor(() => expect(client.imageSearch).toHaveBeenCalledWith(file, "", 12, false));
    expect(await screen.findByRole("heading", { name: "Samsung Wireless Earbuds" })).toBeInTheDocument();
  });

  it("renders saved evaluation metrics", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByText("FULL ML")).toBeInTheDocument();
    expect(screen.getAllByText(/NDCG@10/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MRR/).length).toBeGreaterThan(0);
  });

  it("renders real FULL mode model and vector-index status", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Index Status" }));
    await waitFor(() => expect(client.indexStatus).toHaveBeenCalledTimes(1));
    expect((await screen.findAllByText("FULL ML")).length).toBeGreaterThan(0);
    expect(screen.getByText("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")).toBeInTheDocument();
    expect(screen.getByText("open_clip:ViT-B-32/laion2b_s34b_b79k")).toBeInTheDocument();
    expect(screen.getAllByText("faiss-hnsw-ip-v1").length).toBeGreaterThan(0);
  });
});
