# Chat With Your Docs — RAG App

Upload PDFs or text files and ask questions. Answers come only from your documents and cite the exact passages they used. Click a citation to see the source text highlighted.

Live demo: https://chatwithyourdocs.vercel.app

## Dashboard

| Page | What it shows |
|---|---|
| Overview | Live 5-step pipeline, key stats, answer-time and retrieval-quality charts, chunks per document, recent questions |
| Chat | Streaming answers with clickable citations, a retrieved-passages inspector, and timing for each answer |
| Documents | Table of files with word and chunk counts. Click a file to open its chunk map |
| Retrieval lab | Run a search without calling the AI. Compare semantic, keyword and fused rankings side by side |
| Activity | Log of every question with best match, search time, first-word time, total time, words and citations. Export to CSV |
| Settings | Passages per question, chunk size, overlap, re-index, model and API status, delete data |

## How it works

```
Upload  ->  Chunk  ->  Embed  ->  Retrieve  ->  Answer
browser    browser    browser     browser      server (Claude API)
```

1. **Upload.** PDFs are parsed page by page with pdf.js. Text, Markdown, CSV, JSON and HTML are read directly.
2. **Chunk.** Text splits into ~900-character chunks at sentence or paragraph breaks, with 150 characters of overlap so facts on a boundary are not lost.
3. **Embed.** Each chunk becomes a 384-dimension vector using `all-MiniLM-L6-v2`, running in the browser with Transformers.js. The model (about 23 MB) downloads once and is cached. Vectors are saved in IndexedDB, so reloading the page does not re-embed.
4. **Retrieve.** Hybrid search. The question is embedded and compared to every chunk by cosine similarity (semantic). A BM25 index scores the same chunks by keywords. Reciprocal rank fusion merges both rankings. The top 8 chunks, capped at 30,000 characters, are kept.
5. **Answer.** The question and retrieved passages go to a serverless function, which calls the Claude API and streams the answer back. The system prompt requires citations and forbids outside knowledge. Citations that do not match a sent passage are not linked.

The API key lives only on the server. Documents and vectors stay in the browser. Only the retrieved passages leave the device.

## Tech stack

| Part | Tool |
|---|---|
| Frontend | HTML, CSS, JavaScript (no framework, no build step) |
| PDF parsing | pdf.js 3.11 |
| Embeddings | Transformers.js 3.8, `Xenova/all-MiniLM-L6-v2` |
| Keyword search | BM25 (own implementation) |
| Rank fusion | Reciprocal rank fusion, k = 60 |
| Vector store | In-memory array, persisted to IndexedDB |
| LLM | Claude API (`claude-haiku-4-5-20251001` by default) |
| Backend | Vercel Edge Function |
| Tests | Node test runner |

## Project structure

```
index.html        page layout
styles.css        styles, light and dark themes
app.js            dashboard pages, upload, embedding, search, chat, activity log
rag.js            chunking, tokenizing, BM25, cosine, RRF, hybrid search
api/chat.js       serverless endpoint: GET returns health, POST calls Claude and streams text
tests/            unit tests for retrieval and the stream parser
```

## Run locally

1. Install Node.js 18 or newer.
2. Install the Vercel CLI: `npm i -g vercel`
3. Copy `.env.example` to `.env.local` and add your key from https://console.anthropic.com
4. Run `vercel dev` and open http://localhost:3000

Run tests with `npm test`.

## Deploy (free on Vercel)

1. Push this folder to a new GitHub repository.
2. Go to https://vercel.com, sign in with GitHub, click **Add New > Project**, and import the repository.
3. Leave the framework preset as **Other**. No build command is needed.
4. Under **Environment Variables**, add `ANTHROPIC_API_KEY` with your key.
5. Click **Deploy**. You get a public URL like `your-app.vercel.app`.

**Protect your wallet.** Anyone with the link can ask questions on your key. Set a monthly spend limit in the Anthropic Console. The endpoint also caps each IP at 20 questions per 10 minutes, questions at 2,000 characters, and context at 40,000 characters.

## Cost

Each question sends up to about 8,000 tokens of passages and gets up to 1,024 tokens back. On Claude Haiku 4.5 that is well under one cent per question. Embeddings are free because they run in the visitor's browser. Check current prices at https://www.anthropic.com/pricing

## Limitations

- Scanned PDFs have no text layer and are skipped. OCR is not included.
- Tables in PDFs often extract as jumbled text.
- Search scans every chunk. That is fast for a few thousand chunks, not for millions.
- `all-MiniLM-L6-v2` is trained mostly on English.
- The rate limit is per server instance, so it is a soft limit.
- There is no evaluation set yet.

## Next steps

- Add a reranker to reorder the top 20 before picking 8.
- Build an evaluation set of 30 to 50 questions with known answers, and measure retrieval hit rate and answer accuracy.
- Move embedding to a Web Worker so the page never pauses.
- Add a thumbs up or down on each answer and chart satisfaction on the Overview page.
- Add OCR with Tesseract.js for scanned PDFs.
