import { useEffect, useState } from "react";

import { latestEvaluation } from "../api/client";
import type { EvaluationPayload } from "../types/search";
import { evaluationRows } from "../utils/presentation";

export function EvaluationPage() {
  const [data, setData] = useState<EvaluationPayload | null>(null);

  useEffect(() => {
    latestEvaluation().then(setData).catch(() => setData(null));
  }, []);

  const rows = evaluationRows(data);
  const k = data?.k ?? 10;

  return (
    <main>
      <section className="pageintro">
        <span className="kicker">REAL EVALUATION OUTPUT</span>
        <h1>IR metrics from the labelled demo query set.</h1>
        <p>
          {data?.queries || "—"} curated queries · Recall@{k}, MRR and NDCG@{k}.
          {data?.runtime_mode && <> Runtime artifact: <b>{data.runtime_mode === "full" ? "FULL ML" : "LIGHTWEIGHT"}</b>.</>}
        </p>
      </section>
      <div className="metricgrid">
        {rows.map((row) => (
          <article className="metric" key={row.mode}>
            <span>{row.mode}</span>
            <strong>{row.ndcg.toFixed(3)}</strong>
            <small>NDCG@{k}</small>
            <div><b>{row.recall.toFixed(3)}</b> recall <b>{row.mrr.toFixed(3)}</b> MRR</div>
            <div>p50 {row.p50} ms · p95 {row.p95} ms</div>
          </article>
        ))}
      </div>
      {data?.image_retrieval && (
        <pre className="jsonnote">{JSON.stringify({ image_retrieval: data.image_retrieval }, null, 2)}</pre>
      )}
    </main>
  );
}
