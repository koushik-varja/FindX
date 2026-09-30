import { useState } from "react";

import { Nav } from "./components/Nav";
import { EvaluationPage } from "./pages/EvaluationPage";
import { IndexStatusPage } from "./pages/IndexStatusPage";
import { SearchLabPage } from "./pages/SearchLabPage";
import { SearchPage } from "./pages/SearchPage";
import type { Page } from "./types/search";

export default function App() {
  const [page, setPage] = useState<Page>("search");

  return (
    <>
      <Nav page={page} setPage={setPage} />
      {page === "search" && <SearchPage />}
      {page === "lab" && <SearchLabPage />}
      {page === "evaluation" && <EvaluationPage />}
      {page === "index" && <IndexStatusPage />}
      <footer>FindX · local demo search & ranking system</footer>
    </>
  );
}
