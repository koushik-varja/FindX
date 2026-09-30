import { imageUrl } from "../api/client";
import type { SearchResult } from "../types/search";
import { resultReason } from "../utils/presentation";

export function ResultCard({ result, debug }: { result: SearchResult; debug: boolean }) {
  return (
    <article className="card">
      <img src={imageUrl(result.image_path)} alt={result.title} />
      <div className="cardbody">
        <div className="eyebrow">{result.brand} · {result.category}</div>
        <h3>{result.title}</h3>
        <p>{result.description}</p>
        <div className="price">₹{result.price.toLocaleString("en-IN")}</div>
        <div className="badges">
          <span>{result.colour}</span>
          <span>score {result.score.toFixed(3)}</span>
        </div>
        {debug && (
          <div className="debug">
            <b>{resultReason(result)}</b>
            <code>
              lex {result.debug?.lexical_score?.toFixed?.(3) ?? "-"} · dense {result.debug?.semantic_score?.toFixed?.(3) ?? "-"} · rerank {result.debug?.rerank_score?.toFixed?.(3) ?? "-"}
            </code>
            {(result.debug?.matched_attributes?.length ?? 0) > 0 && <small>{result.debug?.matched_attributes?.join(" · ")}</small>}
          </div>
        )}
      </div>
    </article>
  );
}
