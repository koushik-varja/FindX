import { useState } from "react";

import { imageUrl, searchLab } from "../api/client";
import type { SearchLabPayload } from "../types/search";
import { labModes } from "../utils/presentation";

export function SearchLabPage() {
  const [query, setQuery] = useState("samsoong wirless earbuds under 3000");
  const [data, setData] = useState<SearchLabPayload | null>(null);

  async function run() {
    setData(await searchLab(query));
  }

  return (
    <main>
      <section className="pageintro">
        <span className="kicker">SEARCH LAB</span>
        <h1>Compare ranking stages on the same query.</h1>
        <p>Every column comes from the live backend. No static comparison rows.</p>
        <div className="searchbox">
          <input value={query} onChange={(event) => setQuery(event.target.value)} aria-label="lab query" />
          <button className="primary" onClick={run}>Compare</button>
        </div>
      </section>
      {data && (
        <div className="labgrid">
          {labModes.map((mode) => (
            <section className="labcol" key={mode}>
              <h3>{mode === "bm25" ? "BM25" : mode === "dense" ? "Dense ANN" : mode === "hybrid" ? "Hybrid RRF" : "Reranked"}</h3>
              <small>{data[mode]?.latency_ms?.toFixed(1)} ms · {data[mode]?.runtime_mode === "full" ? "FULL ML" : "LIGHTWEIGHT"}</small>
              {data[mode]?.results?.map((result) => (
                <div className="mini" key={result.id}>
                  <img src={imageUrl(result.image_path)} alt="" />
                  <div>
                    <b>#{result.rank} {result.title}</b>
                    <span>₹{result.price.toLocaleString("en-IN")} · {result.score.toFixed(3)}</span>
                  </div>
                </div>
              ))}
            </section>
          ))}
        </div>
      )}
    </main>
  );
}
