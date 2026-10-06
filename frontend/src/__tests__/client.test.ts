import { afterEach, describe, expect, it, vi } from "vitest";

import { API, imageSearch, searchLab, textSearch } from "../api/client";
import type { SearchPayload } from "../types/search";

const payload: SearchPayload = {
  mode: "image",
  runtime_mode: "full",
  results: [],
};

function successfulResponse(): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("FindX API client routing", () => {
  it("posts text queries and the explicit debug flag", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(successfulResponse());
    vi.stubGlobal("fetch", fetchMock);

    await textSearch("samsoong wirless earbuds", true);

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API}/api/search/text`);
    expect(options?.method).toBe("POST");
    expect(options?.body).toContain("samsoong wirless earbuds");
    expect(options?.body).toContain('"debug":true');
  });

  it("requests debug metadata on the Search Lab developer path", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(successfulResponse());
    vi.stubGlobal("fetch", fetchMock);
    await searchLab("wireless earbuds", true);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API}/api/search/lab`);
    expect(options?.body).toContain('"debug":true');
  });

  it("uses the image endpoint when no refinement text is supplied", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(successfulResponse());
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["image"], "query.png", { type: "image/png" });

    await imageSearch(file);

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API}/api/search/image`);
    expect(options?.method).toBe("POST");
    const form = options?.body as FormData;
    expect(form.get("file")).toBe(file);
    expect(form.get("text")).toBeNull();
    expect(form.get("debug")).toBe("false");
  });

  it("uses the multimodal endpoint and includes refinement text", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(successfulResponse());
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["image"], "query.png", { type: "image/png" });

    await imageSearch(file, "similar but blue", 12, true);

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API}/api/search/multimodal`);
    expect(options?.method).toBe("POST");
    const form = options?.body as FormData;
    expect(form.get("file")).toBe(file);
    expect(form.get("text")).toBe("similar but blue");
    expect(form.get("debug")).toBe("true");
  });
});
