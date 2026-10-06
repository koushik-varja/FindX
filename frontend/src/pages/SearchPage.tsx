import { useRef, useState } from "react";

import { imageSearch, textSearch } from "../api/client";
import { ResultCard } from "../components/ResultCard";
import type { SearchPayload } from "../types/search";
import { correctionLabel, filterChips } from "../utils/presentation";

const suggestions = [
  "samsoong wirless earbuds",
  "3k ke andar black running shoes",
  "काले रनिंग जूते",
  "blue kurti cotton under 1500",
];

type SpeechRecognitionResultEvent = { results: ArrayLike<ArrayLike<{ transcript: string }>> };
type SpeechRecognitionInstance = { lang: string; onresult: ((event: SpeechRecognitionResultEvent) => void) | null; start: () => void };
type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;
type SpeechEnabledWindow = Window & { SpeechRecognition?: SpeechRecognitionConstructor; webkitSpeechRecognition?: SpeechRecognitionConstructor };

export function SearchPage() {
  const [query, setQuery] = useState("black running shoes under 3000");
  const [data, setData] = useState<SearchPayload | null>(null);
  const [debug, setDebug] = useState(false);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function runTextSearch() {
    setLoading(true);
    try { setData(await textSearch(query, debug)); } finally { setLoading(false); }
  }

  async function runImageSearch(multimodal: boolean) {
    if (!file) return;
    setLoading(true);
    try { setData(await imageSearch(file, multimodal ? query : "", 12, debug)); } finally { setLoading(false); }
  }

  function pickFile(next: File | null) {
    setFile(next);
    if (next) setPreview(URL.createObjectURL(next));
  }

  function voiceSearch() {
    const speechWindow = window as SpeechEnabledWindow;
    const SpeechRecognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;
    if (!SpeechRecognition) { alert("Speech recognition is not available in this browser."); return; }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.onresult = (event) => setQuery(event.results[0][0].transcript);
    recognition.start();
  }

  return (
    <main>
      <section className="hero">
        <div className="kicker">MULTIMODAL · MULTILINGUAL · EXPLAINABLE</div>
        <h1>Search by what you mean,<br />not only what you type.</h1>
        <p>Compare lexical, dense and hybrid retrieval; recover common typos; parse commerce constraints; and refine results with an image.</p>
        <div className="searchbox">
          <input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && runTextSearch()} aria-label="search query" />
          <button onClick={voiceSearch} title="Voice search">◉</button>
          <button className="primary" onClick={runTextSearch}>{loading ? "Searching…" : "Search"}</button>
        </div>
        <div className="suggestions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => setQuery(suggestion)}>{suggestion}</button>)}</div>
      </section>

      <section className="imagebar">
        <div>{preview ? <img src={preview} alt="Upload preview" /> : <div className="uploadIcon">＋</div>}<div><b>Image search</b><small>Upload a demo product image or your own product photo.</small></div></div>
        <input ref={inputRef} hidden type="file" accept="image/*" aria-label="product image" onChange={(event) => pickFile(event.target.files?.[0] || null)} />
        <button onClick={() => inputRef.current?.click()}>Choose image</button>
        <button disabled={!file} onClick={() => runImageSearch(false)}>Find similar</button>
        <button className="primary" disabled={!file} onClick={() => runImageSearch(true)}>Image + text</button>
      </section>
<label className="toggle"><input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} /> developer details on next search</label>

      {data && (
        <section className="results">
          <div className="resulthead">
            <div><h2>{data.mode === "image" ? "Visual matches" : "Search results"}</h2><p>{correctionLabel(data.original_query, data.corrected_query)}{data.latency_ms != null && <span> · {data.latency_ms.toFixed(1)} ms</span>}{data.runtime_mode && <span> · {data.runtime_mode === "full" ? "FULL ML" : "LIGHTWEIGHT"}</span>}</p><div className="chips">{filterChips(data.parsed_attributes).map((chip) => <span key={chip}>{chip}</span>)}</div></div>
                      </div>
          {debug && data.timings_ms && <div className="timingstrip">{Object.entries(data.timings_ms).map(([stage, value]) => <span key={stage}><b>{stage}</b> {value.toFixed(2)} ms</span>)}</div>}
          <div className="grid">{data.results.map((result) => <ResultCard key={result.id} result={result} debug={debug} />)}</div>
        </section>
      )}
    </main>
  );
}
