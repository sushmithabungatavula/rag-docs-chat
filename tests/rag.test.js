import test from "node:test";
import assert from "node:assert/strict";
import { chunkText, tokenize, buildBM25, rrf, hybridSearch, searchQuery } from "../rag.js";
import { sseToText } from "../api/chat.js";

test("chunks cover the whole text, stay near target size, and overlap", () => {
  const text = "This is a sentence about apples. ".repeat(200).trim();
  const c = chunkText({ id: "d", name: "d", text });
  assert.equal(c[0].start, 0);
  assert.equal(c.at(-1).end, text.length);
  c.forEach(x => assert.ok(x.text.length <= 1100));
  for (let i = 1; i < c.length; i++) assert.ok(c[i].start < c[i - 1].end, "chunks overlap");
});

test("tokenize drops stopwords and stems", () => {
  assert.deepEqual(tokenize("The runners were running quickly"), ["runner", "runn", "quickly"]);
});

test("BM25 ranks the chunk with the query term first", () => {
  const chunks = [{ text: "cats and dogs" }, { text: "quarterly revenue grew" }, { text: "dogs bark" }];
  const s = buildBM25(chunks).score("revenue");
  assert.equal(s.indexOf(Math.max(...s)), 1);
});

test("RRF rewards items ranked high in both lists", () => {
  const f = rrf([[0, 1, 2], [2, 0, 1]]);
  assert.ok(f.get(0) > f.get(1));
});

test("hybrid search returns top hits within budget", () => {
  const chunks = [
    { text: "The refund window is 30 days.", vec: [1, 0] },
    { text: "Shipping takes 5 days.", vec: [0, 1] },
  ];
  const hits = hybridSearch({ chunks, queryVec: [1, 0], query: "refund policy", bm25: buildBM25(chunks), topK: 1 });
  assert.equal(hits[0].chunk.text, chunks[0].text);
});

test("SSE stream converts to plain text", async () => {
  const ev = o => `event: x\ndata: ${JSON.stringify(o)}\n\n`;
  const raw = ev({ type: "message_start" }) + ev({ type: "content_block_delta", delta: { type: "text_delta", text: "Hello " } }) +
    ev({ type: "content_block_delta", delta: { type: "text_delta", text: "world [1]" } }) + ev({ type: "message_stop" });
  const enc = new TextEncoder().encode(raw);
  const stream = new ReadableStream({ start(c) { c.enqueue(enc.slice(0, 40)); c.enqueue(enc.slice(40)); c.close(); } });
  const out = await new Response(sseToText(stream)).text();
  assert.equal(out, "Hello world [1]");
});

test("chunk size and overlap settings are respected", () => {
  const text = "Word ".repeat(1000).trim();
  const c = chunkText({ id: "d", name: "d", text }, { size: 500, overlap: 0 });
  c.forEach(x => assert.ok(x.text.length <= 700));
  for (let i = 1; i < c.length; i++) assert.ok(c[i].start >= c[i - 1].end - 1);
});

test("follow-up questions carry the previous question into search", () => {
  const prev = "How are travel expenses reimbursed?";
  assert.equal(searchQuery("What about flights?", prev), "What about flights? " + prev);
  assert.equal(searchQuery("Is it paid within a week for everyone?", prev), "Is it paid within a week for everyone? " + prev);
  assert.equal(searchQuery("How many days of paid leave do employees get?", prev), "How many days of paid leave do employees get?");
  assert.equal(searchQuery("What about flights?", ""), "What about flights?");
});
