import test from "node:test";
import assert from "node:assert/strict";
import handler from "../api/chat.js";

const post = (body, ip = "10.0.0." + Math.floor(Math.random() * 250)) =>
  new Request("http://local/api/chat", { method: "POST", headers: { "content-type": "application/json", "x-forwarded-for": ip }, body: JSON.stringify(body) });
const ask = { question: "How long do refunds take?", passages: [{ name: "policy.txt", text: "Refunds take 14 days." }] };

// Replace fetch with a fake Anthropic API that streams `text` and records the request.
function fakeAnthropic(text, { status = 200 } = {}) {
  const calls = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body), headers: init.headers });
    if (status !== 200) return new Response("upstream error", { status });
    const ev = o => `event: ${o.type}\ndata: ${JSON.stringify(o)}\n\n`;
    const sse = ev({ type: "message_start" }) + text.split(" ").map((w, i) =>
      ev({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: (i ? " " : "") + w } })).join("") + ev({ type: "message_stop" });
    return new Response(sse, { headers: { "content-type": "text/event-stream" } });
  };
  return { calls, restore: () => (globalThis.fetch = real) };
}

test("GET reports health and whether a key is configured", async () => {
  delete process.env.ANTHROPIC_API_KEY;
  const r = await handler(new Request("http://local/api/chat"));
  assert.deepEqual(await r.json(), { ok: true, model: "claude-haiku-4-5-20251001", keyConfigured: false });
});

test("POST without a key explains the missing key", async () => {
  delete process.env.ANTHROPIC_API_KEY;
  const r = await handler(post(ask));
  assert.equal(r.status, 500);
  assert.match((await r.json()).error, /ANTHROPIC_API_KEY/);
});

test("rejects empty questions and missing passages", async () => {
  process.env.ANTHROPIC_API_KEY = "test-key";
  assert.equal((await handler(post({ question: " ", passages: ask.passages }))).status, 400);
  assert.equal((await handler(post({ question: "hi", passages: [] }))).status, 400);
});

test("streams the answer and sends numbered passages, system prompt and key", async () => {
  process.env.ANTHROPIC_API_KEY = "test-key";
  const fake = fakeAnthropic("Refunds take 14 days [1].");
  try {
    const r = await handler(post(ask));
    assert.equal(r.status, 200);
    assert.equal(await r.text(), "Refunds take 14 days [1].");
    const { url, body, headers } = fake.calls[0];
    assert.equal(url, "https://api.anthropic.com/v1/messages");
    assert.equal(headers["x-api-key"], "test-key");
    assert.equal(body.stream, true);
    assert.match(body.system, /Cite every claim/);
    assert.match(body.messages.at(-1).content, /\[1\] \(from "policy.txt"\)\nRefunds take 14 days\./);
  } finally { fake.restore(); }
});

test("history keeps alternating turns and ends with the new question", async () => {
  process.env.ANTHROPIC_API_KEY = "test-key";
  const fake = fakeAnthropic("ok");
  try {
    const history = [{ role: "assistant", content: "stray" }, { role: "user", content: "q1" }, { role: "assistant", content: "a1" }, { role: "user", content: "dangling" }];
    await (await handler(post({ ...ask, history }))).text();
    const roles = fake.calls[0].body.messages.map(m => m.role);
    assert.deepEqual(roles, ["user", "assistant", "user"]);
    assert.equal(fake.calls[0].body.messages[0].content, "q1");
  } finally { fake.restore(); }
});

test("upstream errors become friendly messages", async () => {
  process.env.ANTHROPIC_API_KEY = "test-key";
  for (const [status, expect] of [[429, 429], [500, 502]]) {
    const fake = fakeAnthropic("", { status });
    const orig = console.error; console.error = () => {};
    try {
      const r = await handler(post(ask));
      assert.equal(r.status, expect);
      assert.ok((await r.json()).error);
    } finally { fake.restore(); console.error = orig; }
  }
});

test("rate limit blocks the 21st question from one IP", async () => {
  process.env.ANTHROPIC_API_KEY = "test-key";
  const fake = fakeAnthropic("ok");
  try {
    const statuses = [];
    for (let i = 0; i < 21; i++) statuses.push((await handler(post(ask, "192.0.2.7"))).status);
    assert.deepEqual(statuses.slice(0, 20), Array(20).fill(200));
    assert.equal(statuses[20], 429);
  } finally { fake.restore(); }
});
