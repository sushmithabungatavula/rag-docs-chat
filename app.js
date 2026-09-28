const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { chunkText, buildBM25, hybridSearch, CONTEXT_BUDGET } from "./rag.js";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2"; // 384-dimension sentence embeddings, runs in the browser
const DB_NAME = "docs-chat", STORE = "kv";
const DEFAULTS = { topK: 8, chunkSize: 900, overlap: 150 };
const $ = id => document.getElementById(id);
const $$ = sel => [...document.querySelectorAll(sel)];

const state = {
  docs: [],        // {id, name, type, size, text, addedAt}
  chunks: [],      // {docId, docName, start, end, text, vec}
  indexedWith: null,
  settings: { ...DEFAULTS },
  activity: [],    // one row per question
  turns: [],
  cites: {},
  bm25: null,
  api: null,
};
let extractor = null, extractorLoading = null, ctl = null;
const wait = ms => new Promise(r => setTimeout(r, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ms));
const words = t => (t.match(/\S+/g) || []).length;
const fmtS = ms => ms == null ? "" : (ms / 1000).toFixed(1) + " s";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ================= Storage (IndexedDB)
let dbp;
function idb() {
  return dbp ||= new Promise((res, rej) => {
    const r = indexedDB.open(DB_NAME, 2);
    r.onupgradeneeded = () => { if (!r.result.objectStoreNames.contains(STORE)) r.result.createObjectStore(STORE); };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function put(key, value) {
  try { const db = await idb(); db.transaction(STORE, "readwrite").objectStore(STORE).put(value, key); }
  catch { toast("Could not save to browser storage."); }
}
async function get(key) {
  try {
    const db = await idb();
    return await new Promise(res => { const r = db.transaction(STORE).objectStore(STORE).get(key); r.onsuccess = () => res(r.result); r.onerror = () => res(null); });
  } catch { return null; }
}
const saveLibrary = () => put("library", {
  model: MODEL_ID, indexedWith: state.indexedWith, docs: state.docs,
  chunks: state.chunks.map(c => ({ ...c, vec: Array.from(c.vec) })),
});
const saveActivity = () => put("activity", state.activity.slice(-500));
const saveSettings = () => put("settings", state.settings);

// ================= Status and toasts
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), 4500);
}
function setModel(kind, text) { $("modelDot").className = "dot " + kind; $("modelState").textContent = text; renderModelInfo(); }
function step(id, s, text) { const el = $("s-" + id); el.className = "step " + s; if (text !== undefined) el.querySelector("small").textContent = text; }

// ================= Embedding
async function getExtractor() {
  if (extractor) return extractor;
  if (extractorLoading) return extractorLoading;
  setModel("busy", "Loading embedding model");
  const seen = {};
  extractorLoading = import(TRANSFORMERS_URL).then(({ pipeline }) => pipeline("feature-extraction", MODEL_ID, {
    dtype: "q8",
    progress_callback: p => {
      if (p.status === "progress" && p.total) {
        seen[p.file] = [p.loaded, p.total];
        const [l, t] = Object.values(seen).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
        const pct = Math.round(l / t * 100);
        setModel("busy", `Downloading model, ${pct}%`);
        step("embed", "active", `Downloading model, ${pct}%`);
      }
    },
  })).then(ex => { extractor = ex; setModel("ok", "Embedding model ready"); return ex; })
    .catch(e => { extractorLoading = null; setModel("bad", "Embedding model failed to load"); throw e; });
  return extractorLoading;
}
async function embed(texts, onProgress) {
  const ex = await getExtractor();
  const out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const t = await ex(texts.slice(i, i + 16), { pooling: "mean", normalize: true });
    out.push(...t.tolist().map(v => Float32Array.from(v)));
    onProgress?.(Math.min(i + 16, texts.length), texts.length);
    await new Promise(r => setTimeout(r));
  }
  return out;
}

// ================= Indexing
const chunkOpts = () => ({ size: state.settings.chunkSize, overlap: state.settings.overlap });
async function indexDocs(newDocs, { replace = false } = {}) {
  setBusy(true);
  try {
    const base = replace ? [] : state.chunks;
    const docsAfter = replace ? newDocs : [...state.docs, ...newDocs];
    step("upload", "done", `${docsAfter.length} file${docsAfter.length === 1 ? "" : "s"}, ${docsAfter.reduce((s, d) => s + words(d.text), 0).toLocaleString()} words`);
    step("chunk", "active", "Splitting text"); await wait(250);
    const fresh = newDocs.flatMap(d => chunkText(d, chunkOpts()));
    step("chunk", "done", `${base.length + fresh.length} chunks`);
    step("embed", "active", "Preparing");
    const vecs = await embed(fresh.map(c => c.text), (n, t) => step("embed", "active", `Embedding ${n} of ${t}`));
    fresh.forEach((c, i) => (c.vec = vecs[i]));
    state.docs = docsAfter;
    state.chunks = [...base, ...fresh];
    state.indexedWith = chunkOpts();
    state.bm25 = buildBM25(state.chunks);
    await saveLibrary();
    toast(replace ? "Documents re-indexed." : `Added ${newDocs.length} document${newDocs.length === 1 ? "" : "s"}.`);
  } catch (e) {
    console.error(e);
    toast("Indexing failed. The embedding model could not load. Check your connection and try again.");
  } finally {
    setBusy(false);
    renderAll();
  }
}
function setBusy(b) { $("send").disabled = b; $("reindex").disabled = b; $("uploadBtn").disabled = b; }

// ================= Upload
function htmlToText(h) {
  const d = new DOMParser().parseFromString(h, "text/html");
  d.querySelectorAll("script,style").forEach(n => n.remove());
  return d.body.innerText || d.body.textContent || "";
}
async function pdfToText(file) {
  if (!window.pdfjsLib) throw new Error("nopdf");
  const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    step("upload", "active", `Reading ${file.name}, page ${i} of ${pdf.numPages}`);
    const tc = await (await pdf.getPage(i)).getTextContent();
    let out = "";
    tc.items.forEach(it => { out += it.str + (it.hasEOL ? "\n" : " "); });
    pages.push(out.replace(/ +/g, " "));
  }
  return pages.join("\n\n");
}
function newDoc(name, text, type, size) {
  text = text.replace(/\r\n/g, "\n").replace(/[ \t]+\n/g, "\n").trim();
  return text ? { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), name, text, type, size, addedAt: Date.now() } : null;
}
async function addFiles(list) {
  if (!list.length) return;
  const added = [], skipped = [];
  for (const f of list) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    const isPdf = ext === "pdf";
    if (f.size > (isPdf ? 20e6 : 3e6)) { skipped.push(f.name + " (too large)"); continue; }
    if (ext === "doc" || ext === "docx") { skipped.push(f.name + " (save as PDF or .txt first)"); continue; }
    step("upload", "active", "Reading " + f.name);
    try {
      let t = isPdf ? await pdfToText(f) : await f.text();
      if (ext === "html" || ext === "htm") t = htmlToText(t);
      const d = newDoc(f.name, t, ext.toUpperCase(), f.size);
      if (d) added.push(d); else skipped.push(f.name + (isPdf ? " (no text, may be a scan)" : " (empty)"));
    } catch (e) { skipped.push(f.name + (e.message === "nopdf" ? " (PDF reader failed to load)" : " (could not read)")); }
  }
  if (skipped.length) toast("Skipped: " + skipped.join(", "));
  if (added.length) { if (location.hash !== "#documents" && location.hash !== "#overview") location.hash = "#overview"; await indexDocs(added); }
  else renderPipeline();
}
async function removeDoc(id) {
  const d = state.docs.find(x => x.id === id);
  if (!d || !confirm(`Remove "${d.name}"?`)) return;
  state.docs = state.docs.filter(x => x.id !== id);
  state.chunks = state.chunks.filter(c => c.docId !== id);
  state.bm25 = state.chunks.length ? buildBM25(state.chunks) : null;
  await saveLibrary(); renderAll();
}

// ================= Search
async function search(query) {
  const t0 = performance.now();
  const [queryVec] = await embed([query]);
  const t1 = performance.now();
  const hits = hybridSearch({ chunks: state.chunks, queryVec, query, bm25: state.bm25, topK: state.settings.topK, budget: CONTEXT_BUDGET });
  const t2 = performance.now();
  return { hits, embedMs: t1 - t0, searchMs: t2 - t1, totalMs: t2 - t0 };
}

// ================= Chat
function renderAnswer(el, text, map) {
  const lines = esc(text).split("\n");
  let html = "", list = false, para = [];
  const flush = () => { if (para.length) { html += "<p>" + para.join(" ") + "</p>"; para = []; } };
  for (const raw of lines) {
    const line = raw.trim();
    if (/^[-*•]\s+/.test(line)) { flush(); if (!list) { html += "<ul>"; list = true; } html += "<li>" + line.replace(/^[-*•]\s+/, "") + "</li>"; continue; }
    if (list) { html += "</ul>"; list = false; }
    if (!line) { flush(); continue; }
    para.push(line.replace(/^#+\s*/, ""));
  }
  flush(); if (list) html += "</ul>";
  el.innerHTML = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d{1,2})\]/g, (m, n) => map[n] ? `<button class="cite" data-key="${map[n]}" aria-label="Show source ${n}">${n}</button>` : m);
}
function addMsg(role, html) {
  $("intro")?.remove();
  const div = document.createElement("div");
  div.className = "msg " + role; div.innerHTML = html;
  $("threadInner").appendChild(div);
  $("thread").scrollTop = $("thread").scrollHeight;
  return div;
}
function showSource(key) {
  const c = state.cites[key]; if (!c) return;
  const d = state.docs.find(x => x.id === c.docId);
  if (!d) { toast("That document was removed."); return; }
  const from = Math.max(0, c.start - 500), to = Math.min(d.text.length, c.end + 500);
  $("sourceTitle").textContent = `[${c.n}] ${d.name}`;
  $("sourceBody").innerHTML = (from ? "… " : "") + esc(d.text.slice(from, c.start)) + "<mark>" + esc(d.text.slice(c.start, c.end)) + "</mark>" + esc(d.text.slice(c.end, to)) + (to < d.text.length ? " …" : "");
  $("source").hidden = false;
  $("sourceBody").querySelector("mark")?.scrollIntoView({ block: "center" });
}

async function ask() {
  const q = $("q").value.trim();
  if (!q) return;
  if (!state.chunks.length) { toast("Add a document first."); return; }
  $("q").value = ""; autosize(); $("chatStatus").textContent = "";
  addMsg("user", esc(q));
  const bot = addMsg("bot", '<div class="answer"><span class="thinking">Retrieving passages...</span></div>');
  const ansEl = bot.querySelector(".answer");
  $("send").hidden = true; $("stop").hidden = false;
  const log = { id: Date.now(), at: Date.now(), question: q, status: "ok" };
  const tStart = performance.now();

  step("retrieve", "active", "Embedding question"); step("answer", "idle", "Waiting");
  const prev = state.turns.length ? state.turns[state.turns.length - 2].content.slice(0, 200) : "";
  let res;
  try { res = await search((q + " " + prev).trim()); }
  catch {
    ansEl.innerHTML = '<div class="err">Search failed. The embedding model could not load.</div>';
    step("retrieve", "idle", "Failed");
    $("send").hidden = false; $("stop").hidden = true;
    finishLog(log, { status: "search failed" });
    return;
  }
  const { hits } = res;
  log.topScore = hits[0]?.dense ?? 0;
  log.searchMs = res.totalMs;
  log.passages = hits.length;
  step("retrieve", "done", `Top ${hits.length} of ${state.chunks.length}, best ${log.topScore.toFixed(2)}, ${Math.round(res.totalMs)} ms`);

  const stamp = log.id.toString(36), map = {};
  hits.forEach((h, i) => { const key = stamp + "-" + (i + 1); map[i + 1] = key; state.cites[key] = { n: i + 1, docId: h.chunk.docId, start: h.chunk.start, end: h.chunk.end }; });
  const best = Math.max(...hits.map(h => h.dense), 0.01);
  const det = document.createElement("details");
  det.className = "hits";
  det.innerHTML = `<summary>Retrieved ${hits.length} passages in ${Math.round(res.totalMs)} ms</summary>` + hits.map((h, i) =>
    `<button class="hit" data-key="${map[i + 1]}"><span class="n">${i + 1}</span><span class="top"><span class="doc">${esc(h.chunk.docName)}</span><span class="score">${h.dense.toFixed(3)}</span></span><span class="bar"><i style="width:${Math.max(2, Math.round(Math.max(h.dense, 0) / best * 100))}%"></i></span><span class="scores">Semantic ${h.dense.toFixed(3)}, keyword ${h.keyword.toFixed(2)}</span><span class="snip">${esc(h.chunk.text.slice(0, 220))}</span></button>`).join("");
  bot.appendChild(det);

  step("answer", "active", "Waiting for the AI");
  ansEl.innerHTML = '<span class="thinking">Thinking...</span>';
  ctl = new AbortController();
  let text = "";
  try {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "content-type": "application/json" }, signal: ctl.signal,
      body: JSON.stringify({ question: q, passages: hits.map(h => ({ name: h.chunk.docName, text: h.chunk.text })), history: state.turns.slice(-6) }),
    });
    if (!r.ok) { const err = await r.json().catch(() => ({})); throw new Error(err.error || `Request failed (${r.status}).`); }
    const reader = r.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (log.firstTokenMs == null) log.firstTokenMs = performance.now() - tStart;
      text += dec.decode(value, { stream: true });
      renderAnswer(ansEl, text, map);
      step("answer", "active", `Writing, ${words(text)} words`);
      $("thread").scrollTop = $("thread").scrollHeight;
    }
    renderAnswer(ansEl, text, map);
    state.turns.push({ role: "user", content: q }, { role: "assistant", content: text });
    log.cited = new Set((text.match(/\[(\d{1,2})\]/g) || []).map(s => s.slice(1, -1)).filter(n => map[n])).size;
    log.words = words(text);
    step("answer", "done", `${log.words} words, ${log.cited} source${log.cited === 1 ? "" : "s"} cited`);
  } catch (e) {
    const stopped = e.name === "AbortError";
    if (text) renderAnswer(ansEl, text, map); else ansEl.innerHTML = "";
    ansEl.insertAdjacentHTML("beforeend", `<div class="err">${esc(stopped ? "Stopped." : e.message || "The answer failed. Try again.")}</div>`);
    step("answer", "idle", stopped ? "Stopped" : "Failed");
    log.status = stopped ? "stopped" : "failed";
    log.words = words(text);
  } finally {
    $("send").hidden = false; $("stop").hidden = true; ctl = null;
    finishLog(log, { totalMs: performance.now() - tStart });
    const m = document.createElement("div");
    m.className = "metrics";
    m.textContent = `Search ${Math.round(log.searchMs || 0)} ms. First word ${fmtS(log.firstTokenMs) || "none"}. Total ${fmtS(log.totalMs)}.`;
    bot.insertBefore(m, det);
  }
}
function finishLog(log, extra) {
  Object.assign(log, extra);
  if (log.totalMs == null) log.totalMs = 0;
  state.activity.push(log);
  saveActivity();
  renderOverview(); renderActivity();
}

// ================= Rendering: pipeline, stats, charts
function renderPipeline() {
  const n = state.docs.length;
  if (!n) {
    step("upload", "idle", "No files yet"); step("chunk", "idle", "Waiting"); step("embed", "idle", "Waiting");
    step("retrieve", "idle", "Ask a question"); step("answer", "idle", "Ask a question"); return;
  }
  const avg = Math.round(state.chunks.reduce((s, c) => s + c.text.length, 0) / (state.chunks.length || 1));
  step("upload", "done", `${n} file${n === 1 ? "" : "s"}, ${state.docs.reduce((s, d) => s + words(d.text), 0).toLocaleString()} words`);
  step("chunk", "done", `${state.chunks.length} chunks, ~${avg} chars each`);
  step("embed", "done", `${state.chunks.length} vectors, ${state.chunks[0]?.vec.length || 384} dimensions`);
  const last = state.activity.at(-1);
  if (last && last.topScore != null) step("retrieve", "done", `Last: best match ${last.topScore.toFixed(2)}`);
  else step("retrieve", "idle", "Ask a question");
  if (last && last.status === "ok") step("answer", "done", `Last: ${fmtS(last.totalMs)}, ${last.cited ?? 0} cited`);
  else step("answer", "idle", "Ask a question");
}
function stat(v, l, s = "") { return `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`; }
function renderOverview() {
  renderPipeline();
  const ok = state.activity.filter(a => a.status === "ok");
  const avg = arr => arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : null;
  const avgTotal = avg(ok.map(a => a.totalMs));
  const avgSim = avg(state.activity.filter(a => a.topScore != null).map(a => a.topScore));
  const totalWords = state.docs.reduce((s, d) => s + words(d.text), 0);
  $("stats").innerHTML =
    stat(state.docs.length, "Documents", `${totalWords.toLocaleString()} words`) +
    stat(state.chunks.length, "Chunks", `${state.settings.chunkSize} chars, ${state.settings.overlap} overlap`) +
    stat(state.chunks.length ? state.chunks[0].vec.length : 384, "Vector dimensions", "all-MiniLM-L6-v2") +
    stat(state.activity.length, "Questions asked", `${state.activity.length - ok.length} failed or stopped`) +
    stat(avgTotal == null ? "None" : fmtS(avgTotal), "Avg answer time", ok.length ? `First word ${fmtS(avg(ok.map(a => a.firstTokenMs || 0)))}` : "") +
    stat(avgSim == null ? "None" : avgSim.toFixed(2), "Avg best match", "Cosine similarity");

  const has = state.docs.length > 0;
  $("overviewEmpty").hidden = has;
  $("overviewCharts").hidden = !has;
  $("navDocCount").textContent = state.docs.length || "";

  const last = state.activity.slice(-20);
  $("chartLatency").innerHTML = barChart(last.map(a => ({ v: (a.totalMs || 0) / 1000, fail: a.status !== "ok", t: `${a.question}\n${fmtS(a.totalMs)}` })), { unit: " s", digits: 1 });
  $("chartSim").innerHTML = barChart(last.filter(a => a.topScore != null).map(a => ({ v: a.topScore, t: `${a.question}\nBest match ${a.topScore.toFixed(3)}` })), { max: 1, ref: 0.3, digits: 2 });
  const per = state.docs.map(d => ({ name: d.name, n: state.chunks.filter(c => c.docId === d.id).length })).sort((a, b) => b.n - a.n).slice(0, 8);
  const maxN = Math.max(1, ...per.map(p => p.n));
  $("chartDocs").innerHTML = per.length ? `<div class="hbars">${per.map(p => `<div class="hbar"><span class="name" title="${esc(p.name)}">${esc(p.name)}</span><span class="num">${p.n}</span><span class="track"><i style="width:${p.n / maxN * 100}%"></i></span></div>`).join("")}</div>` : '<div class="none">No documents</div>';
  const recent = state.activity.slice(-6).reverse();
  $("recentList").innerHTML = recent.length ? `<ul class="recent">${recent.map(a => `<li><span class="q" title="${esc(a.question)}">${esc(a.question)}</span><span class="m">${a.status === "ok" ? fmtS(a.totalMs) : esc(a.status)}</span></li>`).join("")}</ul>` : '<div class="none">No questions yet. Go to Chat to ask one.</div>';
}
function barChart(items, { max, ref, unit = "", digits = 1 } = {}) {
  if (!items.length) return '<div class="none">No questions yet</div>';
  const top = max ?? Math.max(...items.map(i => i.v), 0.1);
  return `<div class="bars"><span class="max">${top.toFixed(digits)}${unit}</span>${ref != null ? `<span class="ref" style="bottom:${ref / top * 100}%" title="Reference ${ref}"></span>` : ""}${items.map(i => `<span class="b${i.fail ? " fail" : ""}" style="height:${Math.max(2, i.v / top * 100)}%" title="${esc(i.t)}"></span>`).join("")}</div><div class="axis"><span>Older</span><span>Newer</span></div>`;
}

// ================= Rendering: documents
function renderDocs() {
  const tb = $("docTable").querySelector("tbody");
  if (!state.docs.length) { tb.innerHTML = '<tr class="empty"><td colspan="6">No documents yet. Drop files above to add them.</td></tr>'; return; }
  tb.innerHTML = state.docs.map(d => `<tr>
    <td class="name-cell"><button data-open="${d.id}">${esc(d.name)}</button></td>
    <td><span class="tag">${esc(d.type || "TXT")}</span></td>
    <td class="num">${words(d.text).toLocaleString()}</td>
    <td class="num">${state.chunks.filter(c => c.docId === d.id).length}</td>
    <td>${new Date(d.addedAt || Date.now()).toLocaleDateString()}</td>
    <td class="num"><button class="icon-btn" data-remove="${d.id}">Remove</button></td></tr>`).join("");
}
function openDrawer(id) {
  const d = state.docs.find(x => x.id === id); if (!d) return;
  const cs = state.chunks.filter(c => c.docId === id).sort((a, b) => a.start - b.start);
  $("drawerTitle").textContent = d.name;
  $("drawerMeta").textContent = `${words(d.text).toLocaleString()} words, ${cs.length} chunks, avg ${Math.round(cs.reduce((s, c) => s + c.text.length, 0) / (cs.length || 1))} characters`;
  const cap = Math.min(cs.length, 300);
  let html = "";
  for (let i = 0; i < cap; i++) {
    const end = i + 1 < cs.length ? cs[i + 1].start : cs[i].end;
    html += `<span><sup>${i + 1}</sup>${esc(d.text.slice(cs[i].start, Math.max(end, cs[i].start)))}</span>`;
  }
  if (cs.length > cap) html += `<p class="hint">Showing the first ${cap} of ${cs.length} chunks.</p>`;
  $("chunkMap").innerHTML = html;
  $("drawer").hidden = false;
  $("drawerClose").focus();
}

// ================= Rendering: retrieval lab
async function runLab() {
  const q = $("labQ").value.trim();
  if (!q) return;
  if (!state.chunks.length) { toast("Add a document first."); return; }
  $("labRun").disabled = true; $("labMeta").textContent = "Embedding query...";
  try {
    const res = await search(q);
    const r = res.hits.ranks, sent = new Set(res.hits.map(h => h.index));
    const topDense = new Set(r.byDense.slice(0, 8)), topSparse = new Set(r.bySparse.slice(0, 8));
    $("labMeta").textContent = `Embedded the query in ${Math.round(res.embedMs)} ms. Scored ${state.chunks.length} chunks in ${Math.round(res.searchMs)} ms. Outlined items are the ${sent.size} passages the AI would receive.`;
    const item = (i, rank, score) => { const c = state.chunks[i]; const both = topDense.has(i) && topSparse.has(i);
      return `<li class="lab-item${sent.has(i) ? " sent" : ""}"><div class="top"><span class="rk">${rank}</span><span class="doc" title="${esc(c.docName)}">${esc(c.docName)}</span>${both ? '<span class="both">both</span>' : ""}<span class="sc">${score}</span></div><div class="snip">${esc(c.text.slice(0, 260))}</div></li>`; };
    const col = (title, hint, list) => `<div class="lab-col panel"><h3>${title}</h3><p class="hint">${hint}</p>${list.length ? `<ol class="lab-list">${list.join("")}</ol>` : '<div class="none">No matches</div>'}</div>`;
    $("labCols").innerHTML =
      col("Fused ranking", "Reciprocal rank fusion of both lists", r.order.slice(0, 8).map((i, k) => item(i, k + 1, r.fused.get(i).toFixed(4)))) +
      col("Semantic", "Cosine similarity of embeddings", r.byDense.slice(0, 8).map((i, k) => item(i, k + 1, r.dense[i].toFixed(3)))) +
      col("Keyword", "BM25 score", r.bySparse.slice(0, 8).map((i, k) => item(i, k + 1, r.sparse[i].toFixed(2))));
  } catch { $("labMeta").textContent = "Search failed. The embedding model could not load."; }
  finally { $("labRun").disabled = false; }
}

// ================= Rendering: activity
function renderActivity() {
  const rows = [...state.activity].reverse();
  const ok = rows.filter(a => a.status === "ok").length;
  $("actSummary").textContent = rows.length ? `${rows.length} questions, ${ok} answered` : "";
  const tb = $("actTable").querySelector("tbody");
  if (!rows.length) { tb.innerHTML = '<tr class="empty"><td colspan="9">No questions yet.</td></tr>'; return; }
  tb.innerHTML = rows.map(a => `<tr>
    <td>${new Date(a.at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</td>
    <td class="qcell">${esc(a.question)}</td>
    <td class="num">${a.topScore != null ? a.topScore.toFixed(3) : ""}</td>
    <td class="num">${a.searchMs != null ? Math.round(a.searchMs) + " ms" : ""}</td>
    <td class="num">${fmtS(a.firstTokenMs)}</td>
    <td class="num">${fmtS(a.totalMs)}</td>
    <td class="num">${a.words ?? ""}</td>
    <td class="num">${a.cited != null ? `${a.cited} of ${a.passages}` : ""}</td>
    <td><span class="tag${a.status === "ok" ? "" : " fail"}">${a.status === "ok" ? "Answered" : esc(a.status[0].toUpperCase() + a.status.slice(1))}</span></td></tr>`).join("");
}
function exportCsv() {
  if (!state.activity.length) { toast("No activity to export."); return; }
  const cols = ["at", "question", "topScore", "searchMs", "firstTokenMs", "totalMs", "words", "cited", "passages", "status"];
  const cell = v => v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
  const csv = [cols.join(","), ...state.activity.map(a => cols.map(c => cell(c === "at" ? new Date(a.at).toISOString() : typeof a[c] === "number" ? Math.round(a[c] * 1000) / 1000 : a[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `rag-activity-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ================= Rendering: settings
function renderSettings() {
  const s = state.settings;
  $("setTopK").value = s.topK; $("outTopK").textContent = s.topK;
  $("setSize").value = s.chunkSize; $("outSize").textContent = s.chunkSize;
  $("setOverlap").value = s.overlap; $("outOverlap").textContent = s.overlap;
  const stale = state.docs.length && state.indexedWith && (state.indexedWith.size !== s.chunkSize || state.indexedWith.overlap !== s.overlap);
  $("reindex").hidden = !stale;
  $("reindexHint").textContent = stale ? "Your documents were indexed with different chunk settings. Re-index to apply the new ones." : "Changing chunk settings requires re-indexing your documents.";
  renderModelInfo();
}
function renderModelInfo() {
  const rows = [
    ["Embedding model", MODEL_ID],
    ["Embedding status", $("modelState").textContent],
    ["Vector dimensions", "384"],
    ["Search", "Hybrid: cosine similarity + BM25, fused with RRF (k = 60)"],
    ["Answer model", state.api?.model || "Unknown"],
    ["API", state.api ? (state.api.keyConfigured ? "Connected" : "Missing ANTHROPIC_API_KEY") : "Not reachable"],
    ["Context budget", `${CONTEXT_BUDGET.toLocaleString()} characters per question`],
  ];
  $("modelInfo").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
}

function renderAll() { renderOverview(); renderDocs(); renderActivity(); renderSettings(); }

// ================= Routing
const TITLES = { overview: "Overview", chat: "Chat", documents: "Documents", retrieval: "Retrieval lab", activity: "Activity", settings: "Settings" };
function route() {
  const page = (location.hash.slice(1) || "overview");
  const name = TITLES[page] ? page : "overview";
  $$(".page").forEach(p => (p.hidden = p.dataset.page !== name));
  $$("[data-nav]").forEach(a => a.dataset.nav === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  $("pageTitle").textContent = TITLES[name];
  document.title = `${TITLES[name]} · Ask your docs`;
  if (name === "chat") $("q").focus();
}

// ================= Wiring
function autosize() { const t = $("q"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 180) + "px"; }
$("q").addEventListener("input", autosize);
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
$("send").onclick = ask;
$("stop").onclick = () => ctl?.abort();
$("threadInner").addEventListener("click", e => { const b = e.target.closest(".cite,.hit"); if (b) showSource(b.dataset.key); });
$("sourceClose").onclick = () => ($("source").hidden = true);

const pickFiles = () => $("fileInput").click();
$("uploadBtn").onclick = pickFiles;
document.addEventListener("click", e => { if (e.target.closest("[data-action=upload]")) pickFiles(); });
$("drop").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickFiles(); } });
$("fileInput").onchange = e => { addFiles([...e.target.files]); e.target.value = ""; };

let dragDepth = 0;
window.addEventListener("dragenter", e => { if (e.dataTransfer?.types.includes("Files")) { dragDepth++; $("overlay").hidden = false; } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("overlay").hidden = true; } });
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", e => { e.preventDefault(); dragDepth = 0; $("overlay").hidden = true; addFiles([...(e.dataTransfer?.files || [])]); });

$("pasteOpen").onclick = () => $("pasteDlg").showModal();
$("pasteDlg").addEventListener("close", async () => {
  if ($("pasteDlg").returnValue !== "add") return;
  const name = $("pasteName").value.trim() || "Pasted text " + (state.docs.length + 1);
  const d = newDoc(name, $("pasteText").value, "TXT", $("pasteText").value.length);
  $("pasteName").value = ""; $("pasteText").value = "";
  if (!d) { toast("Paste some text first."); return; }
  await indexDocs([d]);
});

$("docTable").addEventListener("click", e => {
  const o = e.target.closest("[data-open]"); if (o) openDrawer(o.dataset.open);
  const r = e.target.closest("[data-remove]"); if (r) removeDoc(r.dataset.remove);
});
$("drawerClose").onclick = () => ($("drawer").hidden = true);
$("drawer").addEventListener("click", e => { if (e.target === $("drawer")) $("drawer").hidden = true; });
document.addEventListener("keydown", e => { if (e.key === "Escape") $("drawer").hidden = true; });

$("labRun").onclick = runLab;
$("labQ").addEventListener("keydown", e => { if (e.key === "Enter") runLab(); });

$("exportCsv").onclick = exportCsv;
$("clearLog").onclick = async () => { if (!state.activity.length || !confirm("Clear the activity log?")) return; state.activity = []; await saveActivity(); renderAll(); };

[["setTopK", "outTopK"], ["setSize", "outSize"], ["setOverlap", "outOverlap"]].forEach(([i, o]) => $(i).addEventListener("input", () => ($(o).textContent = $(i).value)));
$("saveSettings").onclick = async () => {
  state.settings = { topK: +$("setTopK").value, chunkSize: +$("setSize").value, overlap: +$("setOverlap").value };
  await saveSettings(); renderAll(); toast("Settings saved.");
};
$("reindex").onclick = async () => { location.hash = "#overview"; await indexDocs(state.docs, { replace: true }); };
$("deleteAll").onclick = async () => {
  if (!state.docs.length || !confirm("Delete all documents and their vectors from this browser?")) return;
  state.docs = []; state.chunks = []; state.bm25 = null; state.indexedWith = null;
  await saveLibrary(); renderAll(); toast("All documents deleted.");
};
window.addEventListener("hashchange", route);

// ================= Boot
(async () => {
  route();
  const [lib, act, set] = await Promise.all([get("library"), get("activity"), get("settings")]);
  if (set) state.settings = { ...DEFAULTS, ...set };
  if (act) state.activity = act;
  if (lib?.docs?.length && lib.model === MODEL_ID) {
    state.docs = lib.docs;
    state.chunks = lib.chunks.map(c => ({ ...c, vec: Float32Array.from(c.vec) }));
    state.indexedWith = lib.indexedWith || { size: 900, overlap: 150 };
    state.bm25 = buildBM25(state.chunks);
  }
  renderAll();
  fetch("/api/chat").then(r => r.ok ? r.json() : null).catch(() => null).then(info => {
    state.api = info;
    $("apiDot").className = "dot " + (info?.keyConfigured ? "ok" : "bad");
    $("apiState").textContent = info ? (info.keyConfigured ? "API connected" : "API key missing") : "API not reachable";
    renderModelInfo();
  });
})();
