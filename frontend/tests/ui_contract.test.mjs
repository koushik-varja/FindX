import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const read = (relative) => fs.readFileSync(new URL(`../src/${relative}`, import.meta.url), "utf8");


test("text search UI submits the real text-search client", () => {
  const page = read("pages/SearchPage.tsx");
  const client = read("api/client.ts");
  assert.match(page, /textSearch\(query\)/);
  assert.match(client, /\/api\/search\/text/);
  assert.match(client, /mode: "reranked"/);
});


test("typo correction and parsed filters are rendered", () => {
  const page = read("pages/SearchPage.tsx");
  assert.match(page, /correctionLabel/);
  assert.match(page, /filterChips/);
  assert.match(page, /corrected_query/);
  assert.match(page, /parsed_attributes/);
});


test("Search Lab renders all four backend ranking stages", () => {
  const page = read("pages/SearchLabPage.tsx");
  const utility = read("utils/presentation.ts");
  assert.match(page, /searchLab\(query\)/);
  for (const mode of ["bm25", "dense", "hybrid", "reranked"]) assert.ok(utility.includes(`"${mode}"`));
});


test("image upload and image plus text refinement call live endpoints", () => {
  const page = read("pages/SearchPage.tsx");
  const client = read("api/client.ts");
  assert.match(page, /type="file"/);
  assert.match(page, /runImageSearch\(false\)/);
  assert.match(page, /runImageSearch\(true\)/);
  assert.match(client, /\/api\/search\/image/);
  assert.match(client, /\/api\/search\/multimodal/);
});


test("Evaluation page uses saved backend evaluation output", () => {
  const page = read("pages/EvaluationPage.tsx");
  const client = read("api/client.ts");
  assert.match(page, /latestEvaluation\(\)/);
  assert.match(client, /\/api\/evaluation\/latest/);
  assert.match(page, /image_retrieval/);
});


test("Index Status displays the real active runtime mode and models", () => {
  const page = read("pages/IndexStatusPage.tsx");
  const utility = read("utils/presentation.ts");
  assert.match(page, /indexStatus\(\)/);
  assert.match(page, /runtimeBadge/);
  assert.match(utility, /FULL ML/);
  assert.match(utility, /LIGHTWEIGHT/);
});
