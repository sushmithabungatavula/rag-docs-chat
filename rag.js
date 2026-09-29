// Pure retrieval logic: chunking, tokenizing, BM25, cosine, rank fusion.
// No browser APIs here, so it runs in Node tests too.

export const CHUNK_SIZE = 900;
export const CHUNK_OVERLAP = 150;
export const TOP_K = 8;
export const CONTEXT_BUDGET = 30000;

const STOP = new Set(("a an the and or but if of to in on at by for with from as is are was were be been being it its this that " +
  "these those i you he she we they them his her our your their not no do does did have has had can could will would should " +
  "may might must so than then there here what which who whom when where why how all any each more most other some such only " +
  "own same too very just also into over under about after before between out up down").split(" "));

// Split text into ~CHUNK_SIZE pieces, cutting at sentence or paragraph ends, with overlap.
export function chunkText(doc, { size = CHUNK_SIZE, overlap = CHUNK_OVERLAP } = {}) {
  const CHUNK = size, OVER = Math.min(overlap, Math.floor(size / 2));
  const t = doc.text, out = [];
  let start = 0;
  while (start < t.length) {
    let end = Math.min(t.length, start + CHUNK);
    if (end < t.length) {
      const from = start + Math.floor(CHUNK * 0.6);
      const win = t.slice(from, end + 200);
      const m = [...win.matchAll(/\n\s*\n|[.!?]\s/g)].pop();
      if (m) end = from + m.index + m[0].length;
    }
    out.push({ docId: doc.id, docName: doc.name, start, end, text: t.slice(start, end) });
    if (end >= t.length) break;
    let next = Math.max(end - OVER, start + 1);
    const sp = t.indexOf(" ", next);
    if (sp > -1 && sp < end) next = sp + 1;
    start = next;
  }
  return out;
}

function stem(w) { return w.length > 4 ? w.replace(/(ing|edly|ed|ies|es|s)$/, "") : w; }
export function tokenize(s) {
  return (s.toLowerCase().match(/[a-z0-9\u00c0-\u024f]+/g) || [])
    .filter(w => w.length > 1 && !STOP.has(w)).map(stem);
}

// Okapi BM25 keyword index.
export function buildBM25(chunks, k1 = 1.2, b = 0.75) {
  const docs = chunks.map(c => {
    const tf = new Map();
    const toks = tokenize(c.text);
    toks.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
    return { tf, len: toks.length };
  });
  const N = docs.length || 1;
  const avg = docs.reduce((s, d) => s + d.len, 0) / N || 1;
  const df = new Map();
  docs.forEach(d => d.tf.forEach((_, t) => df.set(t, (df.get(t) || 0) + 1)));
  const idf = t => { const n = df.get(t) || 0; return Math.log(1 + (N - n + 0.5) / (n + 0.5)); };
  return {
    vocabSize: df.size,
    score(query) {
      const q = [...new Set(tokenize(query))];
      return docs.map(d => q.reduce((s, t) => {
        const f = d.tf.get(t);
        return f ? s + idf(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.len / avg)) : s;
      }, 0));
    },
  };
}

// Vectors from the embedding model are already L2-normalized, so cosine = dot product.
export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Reciprocal rank fusion: merge rankings without tuning score scales.
export function rrf(rankings, k = 60) {
  const fused = new Map();
  rankings.forEach(list => list.forEach((idx, rank) => fused.set(idx, (fused.get(idx) || 0) + 1 / (k + rank + 1))));
  return fused;
}

// Score every chunk both ways and fuse the rankings.
export function rankAll({ chunks, queryVec, query, bm25 }) {
  const dense = chunks.map(c => cosine(queryVec, c.vec));
  const sparse = bm25.score(query);
  const byDense = dense.map((s, i) => i).sort((a, b) => dense[b] - dense[a]);
  const bySparse = sparse.map((s, i) => i).filter(i => sparse[i] > 0).sort((a, b) => sparse[b] - sparse[a]);
  const fused = rrf([byDense, bySparse]);
  const order = [...fused.keys()].sort((a, b) => fused.get(b) - fused.get(a));
  return { dense, sparse, byDense, bySparse, fused, order };
}

// Follow-ups like "what about flights?" or "is it paid?" need the previous question
// to find the right passages. Standalone questions are searched as they are.
const FOLLOW_START = /^(what about|how about|and|also|what else|why|how come|more|then)\b/i;
const PRONOUN = /\b(it|its|that|this|these|those|they|them|their|there|he|she|him|her)\b/i;
export function searchQuery(question, previous) {
  const q = question.trim();
  if (!previous) return q;
  const short = (q.match(/\S+/g) || []).length <= 5;
  return short || FOLLOW_START.test(q) || PRONOUN.test(q) ? `${q} ${previous.slice(0, 200)}` : q;
}

// Hybrid search: dense (semantic) + BM25 (keyword), fused with RRF, trimmed to topK and a character budget.
export function hybridSearch({ chunks, queryVec, query, bm25, topK = TOP_K, budget = CONTEXT_BUDGET }) {
  const r = rankAll({ chunks, queryVec, query, bm25 });
  const hits = [];
  let used = 0;
  for (const i of r.order) {
    if (hits.length >= topK) break;
    if (used + chunks[i].text.length > budget) continue;
    hits.push({ index: i, chunk: chunks[i], dense: r.dense[i], keyword: r.sparse[i], fused: r.fused.get(i) });
    used += chunks[i].text.length;
  }
  hits.ranks = r;
  return hits;
}
