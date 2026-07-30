const toggleBtn = document.getElementById("chat-toggle");
const panel = document.getElementById("chat-panel");
const closeBtn = document.getElementById("chat-close");
const modelSelect = document.getElementById("chat-model-select");
const statusEl = document.getElementById("chat-status");
const logEl = document.getElementById("chat-log");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");
const suggestionsEl = document.getElementById("chat-suggestions");

const worker = new Worker(new URL("./chat-worker.js", import.meta.url), { type: "module" });

let ready = false;
let loading = false;
let currentModelKey = null;
let assistantDiv = null;
const history = [];

// A 135M model can't usefully attend to the whole guide, so instead of
// dumping all of it in, pick the single section that best matches the
// question (simple keyword overlap) and ground on just that. Shorter
// context and a plainer instruction both raise the odds of a coherent
// answer, per this guide's own advice for small local models.
let sectionsCache = null;

function getSections() {
  if (sectionsCache) return sectionsCache;
  const sections = [];
  document.querySelectorAll(".wrap > section").forEach((section) => {
    const heading = section.querySelector("h2");
    if (!heading) return;
    const text = section.innerText.replace(/\n{3,}/g, "\n\n").trim();
    sections.push({ title: heading.textContent.trim(), text: text.slice(0, 900) });
  });
  sectionsCache = sections;
  return sections;
}

function bestSection(question) {
  const words = question.toLowerCase().match(/[a-z0-9]+/g) || [];
  const sections = getSections();
  let best = sections[0];
  let bestScore = -1;
  for (const s of sections) {
    const lower = s.text.toLowerCase();
    const score = words.reduce((acc, w) => acc + (w.length > 2 && lower.includes(w) ? 1 : 0), 0);
    if (score > bestScore) {
      bestScore = score;
      best = s;
    }
  }
  return best;
}

function systemPrompt(question) {
  const section = bestSection(question || "");
  return (
    "You answer questions about a guide called \"Local LLM Best Practices\". " +
    "Use only the text below. Keep your answer to 2-3 short sentences.\n\n" +
    "Section: " + section.title + "\n" + section.text
  );
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = "chat-msg " + role;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function setBusy(busy) {
  loading = busy;
  sendBtn.disabled = busy || !ready;
  inputEl.disabled = busy || !ready;
}

function loadModel(modelKey) {
  if (loading || modelKey === currentModelKey) return;
  ready = false;
  setBusy(true);
  currentModelKey = modelKey;
  history.length = 0;
  logEl.innerHTML = "";
  statusEl.textContent = "loading " + modelKey + "...";
  worker.postMessage({ type: "load", data: modelKey });
}

function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !ready || loading) return;

  inputEl.value = "";
  suggestionsEl.classList.add("hidden");
  addMessage("user", text);
  history.push({ role: "user", content: text });
  setBusy(true);

  assistantDiv = addMessage("assistant", "...");
  worker.postMessage({
    type: "generate",
    data: { messages: [{ role: "system", content: systemPrompt(text) }, ...history] },
  });
}

worker.addEventListener("message", ({ data: { status, data, output, tps, numTokens, loaded, total } }) => {
  switch (status) {
    case "model_list": {
      modelSelect.innerHTML = "";
      for (const m of data) {
        const opt = document.createElement("option");
        opt.value = m.key;
        opt.textContent = m.label;
        modelSelect.appendChild(opt);
      }
      const preferred = data[0];
      if (preferred) {
        modelSelect.value = preferred.key;
        loadModel(preferred.key);
      }
      break;
    }
    case "loading":
      statusEl.textContent = data;
      break;
    case "progress": {
      const pct = total ? Math.round((100 * (loaded || 0)) / total) : null;
      statusEl.textContent = pct !== null ? `downloading model weights... ${pct}%` : "downloading model weights...";
      break;
    }
    case "ready":
      ready = true;
      setBusy(false);
      suggestionsEl.classList.remove("hidden");
      suggestionsEl.querySelectorAll(".suggestion-chip").forEach((chip) => { chip.disabled = false; });
      addMessage(
        "system-note",
        "AI chatbot ready. It runs locally in your browser, may be wrong, and is not a human. Answers are grounded in this page's content where possible."
      );
      statusEl.textContent = "ready: " + currentModelKey;
      inputEl.focus();
      break;
    case "start":
      if (assistantDiv) assistantDiv.textContent = "";
      break;
    case "update":
      if (assistantDiv) {
        assistantDiv.textContent += output;
        logEl.scrollTop = logEl.scrollHeight;
      }
      break;
    case "complete":
      if (assistantDiv) assistantDiv.textContent = output;
      history.push({ role: "assistant", content: output });
      setBusy(false);
      inputEl.focus();
      break;
    case "error":
      console.error(data);
      if (ready) {
        if (assistantDiv) assistantDiv.textContent = "Error generating a response: " + data;
      } else {
        statusEl.textContent = "failed to load model: " + data;
        currentModelKey = null;
      }
      setBusy(false);
      break;
  }
});

toggleBtn.addEventListener("click", () => {
  const isOpen = panel.classList.toggle("open");
  if (isOpen && modelSelect.options.length === 0) {
    worker.postMessage({ type: "list_models" });
  }
});

closeBtn.addEventListener("click", () => {
  panel.classList.remove("open");
});

modelSelect.addEventListener("change", () => {
  loadModel(modelSelect.value);
});

sendBtn.addEventListener("click", sendMessage);

suggestionsEl.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (chip.disabled || !ready || loading) return;
    inputEl.value = chip.textContent;
    sendMessage();
  });
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + "px";
});
