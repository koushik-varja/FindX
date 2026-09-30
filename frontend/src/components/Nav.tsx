import type { Page } from "../types/search";

const links: Array<[Page, string]> = [
  ["search", "Search"],
  ["lab", "Search Lab"],
  ["evaluation", "Evaluation"],
  ["index", "Index Status"],
];

export function Nav({ page, setPage }: { page: Page; setPage: (page: Page) => void }) {
  return (
    <header>
      <div className="brand">
        <span className="brandmark">F</span>
        <div>
          <b>FindX</b>
          <small>Search & ranking lab</small>
        </div>
      </div>
      <nav>
        {links.map(([target, label]) => (
          <button key={target} className={page === target ? "active" : ""} onClick={() => setPage(target)}>
            {label}
          </button>
        ))}
      </nav>
    </header>
  );
}
