import { useEffect, useState } from "react";

import { indexStatus } from "../api/client";
import type { IndexStatus } from "../types/search";
import { runtimeBadge } from "../utils/presentation";

export function IndexStatusPage() {
  const [data, setData] = useState<IndexStatus | null>(null);

  useEffect(() => {
    indexStatus().then(setData).catch(() => setData(null));
  }, []);

  return (
    <main>
      <section className="pageintro">
        <span className="kicker">INDEX STATUS</span>
        <h1>What is actually loaded.</h1>
        <p>Runtime mode, model names, vector backends and dimensions come directly from the backend.</p>
        {data && <div className="modepill">{runtimeBadge(data)}</div>}
      </section>
      <div className="statusgrid">
        {data && Object.entries(data).map(([key, value]) => (
          <div key={key}>
            <span>{key.replaceAll("_", " ")}</span>
            <b>{String(value)}</b>
          </div>
        ))}
      </div>
    </main>
  );
}
