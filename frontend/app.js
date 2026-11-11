/* =====================================================================
   RAG Studio - frontend logic

   1. Settings        - where the backend lives, mock on/off
   2. API             - the ONLY place that talks to the backend
   3. Config page     - files, form, live pipeline summary, build
   4. Chat page       - sending questions, showing answers + sources
   5. Helpers
   ===================================================================== */

// ---------- 1. Settings ----------
const API_BASE = "http://localhost:8000";
const USE_MOCK = true;   // true = fake responses (no backend needed yet). Set to false once the backend runs.

const state = {
  files: [],          // the PDFs the user picked
  config: null,       // the pipeline settings, saved when "Build" is clicked
  pipelineId: null,   // returned by the backend after building
  numChunks: null,
  busy: false,        // true while waiting for an answer
};

const $ = (selector) => document.querySelector(selector);


// ---------- 2. API ----------
// Backend contract (we'll build this next):
//   POST /api/pipeline  form-data: files[], config (JSON)  -> { pipeline_id, num_chunks }
//   POST /api/chat      JSON: { pipeline_id, question }     -> { answer, sources: [{ source, page, text, score, rerank_score }] }
const api = {
  async buildPipeline(files, config) {
    if (USE_MOCK) return mockBuild(files);
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    body.append("config", JSON.stringify(config));
    return request("/api/pipeline", { method: "POST", body });
  },

  async ask(question) {
    if (USE_MOCK) return mockAsk();
    return request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pipeline_id: state.pipelineId, question }),
    });
  },
};

async function request(path, options) {
  const response = await fetch(API_BASE + path, options);
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try { message = JSON.parse(text).detail || text; } catch { /* plain-text error */ }
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json();
}

// Fake responses so the UI can be shown before the backend exists
async function mockBuild(files) {
  await sleep(2600);
  return { pipeline_id: "mock", num_chunks: files.length * 24 };
}

async function mockAsk() {
  await sleep(1100);
  const source = state.files[0]?.name || "document.pdf";
  const reranked = state.config.reranking.enabled;
  return {
    answer: "This is a mock answer - the backend isn't connected yet. Once it is, the answer will be written from your document and cite pages like this (page 3) and this (page 5).",
    sources: [3, 5, 6].map((page, i) => ({
      source,
      page,
      text: `Preview of the chunk from page ${page}. The real chunk text appears here once the backend is running.`,
      score: 0.82 - i * 0.05,
      rerank_score: reranked ? 0.91 - i * 0.17 : null,
    })),
  };
}


// ---------- 3. Config page ----------

// Read every setting from the form into one plain object (this is what the backend receives)
function readConfig() {
  const form = $("#configForm");
  const data = new FormData(form);
  return {
    chunking: {
      strategy: data.get("chunking"),
      chunk_size: Number(data.get("chunk_size")),
      overlap: Number(data.get("overlap")),
    },
    embedding: { model: data.get("embedding") },
    retrieval: { top_k: Number(data.get("top_k")) },
    reranking: { enabled: form.rerank.checked, model: "rerank-v3.5", top_n: Number(data.get("top_n")) },
    generation: { model: data.get("llm") },
  };
}

// --- Files: pick, drag & drop, remove ---
function setupFiles() {
  const input = $("#fileInput");
  const zone = $("#dropzone");

  input.addEventListener("change", () => {
    addFiles(input.files);
    input.value = "";   // allows picking the same file again after removing it
  });

  ["dragenter", "dragover"].forEach((name) =>
    zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) =>
    zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  $("#fileList").addEventListener("click", (e) => {
    const button = e.target.closest(".remove");
    if (!button) return;
    state.files.splice(Number(button.dataset.index), 1);
    renderFiles();
    renderSummary();
  });
}

function addFiles(fileList) {
  for (const file of fileList) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast(`${file.name} is not a PDF`);
      continue;
    }
    const duplicate = state.files.some((f) => f.name === file.name && f.size === file.size);
    if (!duplicate) state.files.push(file);
  }
  renderFiles();
  renderSummary();
}

function renderFiles() {
  $("#fileList").innerHTML = state.files.map((file, i) => `
    <li class="file">
      <span class="file-icon">PDF</span>
      <span class="file-name">${escapeHtml(file.name)}</span>
      <span class="file-size">${formatSize(file.size)}</span>
      <button type="button" class="remove" data-index="${i}" aria-label="Remove ${escapeHtml(file.name)}">×</button>
    </li>`).join("");
}

// --- Keep the form in sync: re-run every time any setting changes ---
function renderSummary() {
  // "keep best" can't be more than the number of chunks we retrieve
  $("#topN").max = Math.min(10, Number($("#topK").value));

  const c = readConfig();

  $("#topKOut").textContent = c.retrieval.top_k;
  $("#charSettings").hidden = c.chunking.strategy !== "character-wise";
  $("#topN").disabled = !c.reranking.enabled;
  $("#rerankSettings").classList.toggle("off", !c.reranking.enabled);

  const n = state.files.length;
  $("#buildBtn").disabled = n === 0;
  $("#buildHint").textContent = n === 0
    ? "Upload at least one PDF to continue."
    : `${n} PDF${n > 1 ? "s" : ""} ready.`;
}

// --- Build: send files + config, show progress, then switch to chat ---
const BUILD_STEPS = ["Reading PDFs", "Chunking text", "Creating embeddings", "Storing in ChromaDB"];

async function buildPipeline() {
  const config = readConfig();
  if (config.chunking.strategy === "character-wise" && config.chunking.overlap >= config.chunking.chunk_size) {
    toast("Overlap must be smaller than the chunk size");
    return;
  }
  const list = $("#progressList");
  list.innerHTML = BUILD_STEPS.map((step) => `<li>${step}</li>`).join("");
  const items = [...list.children];
  $("#buildOverlay").hidden = false;

  // Move through the steps on a timer while we wait for the backend
  let current = 0;
  items[0].className = "active";
  const timer = setInterval(() => {
    if (current < items.length - 1) {
      items[current].className = "done";
      items[++current].className = "active";
    }
  }, 650);

  try {
    const result = await api.buildPipeline(state.files, config);
    clearInterval(timer);
    items.forEach((li) => (li.className = "done"));
    await sleep(450);

    state.config = config;
    state.pipelineId = result.pipeline_id;
    state.numChunks = result.num_chunks;
    showView("chat");
  } catch (err) {
    clearInterval(timer);
    toast(`Build failed: ${err.message}`);
  } finally {
    $("#buildOverlay").hidden = true;
  }
}

function showView(name) {
  const isChat = name === "chat";
  $("#configView").hidden = isChat;
  $("#chatView").hidden = !isChat;

  $("#stepConfig").className = isChat ? "step done" : "step active";
  $("#stepChat").className = isChat ? "step active" : "step";

  const pill = $("#statusPill");
  pill.textContent = state.pipelineId ? "Pipeline ready" : "Not built";
  pill.classList.toggle("ready", Boolean(state.pipelineId));

  if (isChat) {
    renderSidebar();
    $("#question").focus();
  }
  window.scrollTo(0, 0);
}


// ---------- 4. Chat page ----------
function renderSidebar() {
  $("#sideDocs").innerHTML = state.files.map((file) =>
    `<li><span class="file-icon">PDF</span><span>${escapeHtml(file.name)}</span></li>`).join("");

  const c = state.config;
  const rows = [
    ["Chunking", c.chunking.strategy],
    ["Chunks", state.numChunks ?? "-"],
    ["Embedding", c.embedding.model],
    ["Retrieve", `top ${c.retrieval.top_k}`],
    ["Rerank", c.reranking.enabled ? `best ${c.reranking.top_n}` : "off"],
    ["LLM", c.generation.model],
  ];
  $("#sideConfig").innerHTML = rows.map(([key, value]) =>
    `<div><dt>${key}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join("");
}

async function sendQuestion(text) {
  const question = text.trim();
  if (!question || state.busy) return;

  state.busy = true;
  $("#sendBtn").disabled = true;
  $("#emptyState").hidden = true;
  $("#question").value = "";
  autoGrow();

  appendMessage(`<div class="msg user"><div class="bubble">${escapeHtml(question)}</div></div>`);
  const typing = appendMessage(`
    <div class="msg bot"><div class="avatar">◆</div>
      <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
    </div>`);

  try {
    const result = await api.ask(question);
    typing.remove();
    appendMessage(answerHtml(result));
  } catch (err) {
    typing.remove();
    appendMessage(`<div class="msg bot error"><div class="avatar">!</div><div class="bubble">${escapeHtml(err.message)}</div></div>`);
  } finally {
    state.busy = false;
    $("#sendBtn").disabled = false;
    $("#question").focus();
  }
}

// Turn "(page 5)" in the answer into a small citation chip, and list the sources below
function answerHtml({ answer, sources = [] }) {
  const body = escapeHtml(answer)
    .replace(/\(page (\d+)\)/gi, '<span class="cite">p. $1</span>')
    .replace(/\n/g, "<br>");

  const items = sources.map((s) => `
    <li class="source">
      <div class="source-head">
        <span class="cite">p. ${escapeHtml(String(s.page))}</span>
        <span class="source-name">${escapeHtml(s.source)}</span>
        ${scoreHtml("vector", s.score)}
        ${scoreHtml("rerank", s.rerank_score)}
      </div>
      <p>${escapeHtml(truncate(s.text, 280))}</p>
    </li>`).join("");

  const sourcesBlock = sources.length
    ? `<details class="sources"><summary>${sources.length} source${sources.length > 1 ? "s" : ""}</summary><ol>${items}</ol></details>`
    : "";

  return `<div class="msg bot"><div class="avatar">◆</div><div class="bubble"><div class="answer">${body}</div>${sourcesBlock}</div></div>`;
}

function scoreHtml(label, value) {
  if (value === null || value === undefined) return "";
  return `<span class="score">${label} <b>${Number(value).toFixed(2)}</b></span>`;
}

function appendMessage(html) {
  const holder = document.createElement("div");
  holder.innerHTML = html.trim();
  const element = holder.firstElementChild;
  $("#messages").appendChild(element);
  element.scrollIntoView({ behavior: "smooth", block: "end" });
  return element;
}

function clearChat() {
  document.querySelectorAll("#messages .msg").forEach((el) => el.remove());
  $("#emptyState").hidden = false;
}

// Textarea grows with its content (up to the CSS max-height)
function autoGrow() {
  const box = $("#question");
  box.style.height = "auto";
  box.style.height = `${box.scrollHeight}px`;
}


// ---------- 5. Helpers ----------
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max).trimEnd()}…` : text;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let toastTimer;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}


// ---------- Wire everything up ----------
function init() {
  setupFiles();

  const form = $("#configForm");
  form.addEventListener("input", renderSummary);
  form.addEventListener("change", renderSummary);
  form.addEventListener("submit", (e) => e.preventDefault());
  $("#buildBtn").addEventListener("click", buildPipeline);

  $("#editConfigBtn").addEventListener("click", () => showView("config"));
  $("#clearChatBtn").addEventListener("click", clearChat);

  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    sendQuestion($("#question").value);
  });
  $("#question").addEventListener("input", autoGrow);
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion($("#question").value);
    }
  });
  document.querySelectorAll(".chip[data-q]").forEach((chip) =>
    chip.addEventListener("click", () => sendQuestion(chip.dataset.q)));

  renderSummary();
}

init();
