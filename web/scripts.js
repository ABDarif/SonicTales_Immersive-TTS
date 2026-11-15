"use strict";

const $ = (id) => document.getElementById(id);
const state = { activeJob: null, pollTimer: null, tab: "file", engines: null };

// ---------------------------------------------------------------- helpers

async function api(path, options = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...options });
    let body = null;
    try { body = await res.json(); } catch (_) { /* not JSON */ }
    if (!res.ok) {
        let detail = body && body.detail;
        if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
        const err = new Error(detail || `Request failed (${res.status})`);
        err.status = res.status;
        throw err;
    }
    return body;
}

function show(el, visible) { el.hidden = !visible; }
function setError(el, message) { el.textContent = message || ""; show(el, Boolean(message)); }

function fmtDuration(seconds) {
    if (!seconds) return "";
    const m = Math.floor(seconds / 60), s = Math.round(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
}

// ---------------------------------------------------------------- history

async function loadHistory() {
    const jobs = await api("/api/jobs");
    const list = $("historyList");
    list.replaceChildren();
    show($("historyEmpty"), jobs.length === 0);
    for (const job of jobs) {
        const li = document.createElement("li");
        if (state.activeJob === job.id) li.classList.add("active");
        const row = document.createElement("div");
        row.className = "row";
        const title = document.createElement("span");
        title.className = "title";
        title.textContent = job.title;
        const del = document.createElement("button");
        del.className = "ghost";
        del.textContent = "✕";
        del.title = "Delete";
        del.setAttribute("aria-label", `Delete ${job.title}`);
        del.addEventListener("click", async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete "${job.title}"?`)) return;
            try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
            if (state.activeJob === job.id) { stopPolling(); show($("jobCard"), false); state.activeJob = null; }
            loadHistory();
        });
        row.append(title, del);
        const meta = document.createElement("span");
        meta.className = `meta status-${job.status}`;
        const when = new Date(job.created_at * 1000).toLocaleString();
        meta.textContent = `${job.status}${job.duration_s ? " · " + fmtDuration(job.duration_s) : ""} · ${when}`;
        li.append(row, meta);
        li.addEventListener("click", () => openJob(job.id));
        list.append(li);
    }
}

// ---------------------------------------------------------------- creating a job

function setTab(tab) {
    state.tab = tab;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    show($("paneFile"), tab === "file");
    show($("paneText"), tab === "text");
}

async function loadEngines() {
    try {
        state.engines = await api("/api/engines");
    } catch (_) { return; }
    const select = $("ttsSelect");
    for (const [name, ok] of Object.entries(state.engines.engines)) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = !ok ? `${name} (not available)` : name === "synthetic" ? "synthetic (placeholder tones, not speech)" : name;
        opt.disabled = !ok;
        select.append(opt);
    }
    const real = ["kokoro", "sapi", "openai", "espeak"].some((n) => state.engines.engines[n]);
    if (!real) {
        $("engineNote").textContent =
            "No speech engine is available on this server, so generating will fail until one is installed: " +
            "espeak-ng or Kokoro (see README); Windows has built-in voices. The 'synthetic' option makes placeholder tones only, not speech.";
        show($("engineNote"), true);
    }
    if (!state.engines.builders.llm) {
        const opt = $("builderSelect").querySelector('[value="llm"]');
        opt.disabled = true;
        opt.textContent = "AI (Claude): server key not set";
    }
}

async function createJob() {
    setError($("createError"), "");
    const form = new FormData();
    if (state.tab === "file") {
        const file = $("fileInput").files[0];
        if (!file) { setError($("createError"), "Choose a PDF or TXT file first."); return; }
        form.append("file", file);
    } else {
        const text = $("textInput").value;
        if (!text.trim()) { setError($("createError"), "Paste some text first."); return; }
        form.append("text", text);
    }
    form.append("title", $("titleInput").value);
    form.append("chapters", $("chaptersInput").value);
    form.append("tts", $("ttsSelect").value);
    form.append("builder", $("builderSelect").value);
    form.append("fmt", $("fmtSelect").value);
    form.append("ambience", $("ambienceCheck").checked ? "true" : "false");
    form.append("ambience_db", $("ambienceRange").value);

    $("createBtn").disabled = true;
    try {
        const { id } = await api("/api/jobs", { method: "POST", body: form });
        await loadHistory();
        openJob(id);
    } catch (err) {
        setError($("createError"), err.message);
    } finally {
        $("createBtn").disabled = false;
    }
}

// ---------------------------------------------------------------- job view

function stopPolling() { clearTimeout(state.pollTimer); state.pollTimer = null; }

async function openJob(id) {
    stopPolling();
    state.activeJob = id;
    show($("jobCard"), true);
    $("jobCard").scrollIntoView({ behavior: "smooth", block: "start" });
    await refreshJob(id);
    loadHistory();
}

async function refreshJob(id) {
    if (state.activeJob !== id) return;
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch (err) { setError($("jobError"), err.message); return; }
    $("jobTitle").textContent = job.title;
    $("jobProgress").value = Math.round(job.progress * 100);
    $("jobMessage").textContent = job.status === "done" ? `Ready · ${fmtDuration(job.duration_s)}` : job.message || job.status;
    setError($("jobError"), job.status === "failed" ? job.error || "Generation failed" : "");
    show($("jobProgress"), job.status === "queued" || job.status === "running");
    show($("result"), job.status === "done");
    if (job.status === "done") { await renderResult(job); loadHistory(); return; }
    if (job.status === "queued" || job.status === "running") {
        state.pollTimer = setTimeout(() => refreshJob(id), 1500);
    } else {
        loadHistory();
    }
}

async function renderResult(job) {
    const base = `/api/jobs/${job.id}/audio`;
    const player = $("player");
    if (player.dataset.src !== base) { player.src = base; player.dataset.src = base; }
    $("downloadLink").href = `${base}?download=true`;

    const list = $("chapterList");
    list.replaceChildren();
    for (const ch of job.chapters) {
        if (job.chapters.length < 2 || job.fmt === "wav") break;  // per-chapter files only exist for multi-chapter books
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.textContent = `▶ ${ch.title}`;
        btn.addEventListener("click", () => {
            player.src = `${base}?chapter=${ch.index}`;
            player.dataset.src = player.src;
            player.play().catch(() => {});
        });
        li.append(btn);
        list.append(li);
    }

    try {
        const script = await api(`/api/jobs/${job.id}/script`);
        renderScript(script);
    } catch (_) { $("scriptView").replaceChildren(); }
}

function renderScript(script) {
    const root = $("scriptView");
    root.replaceChildren();
    const cast = document.createElement("p");
    cast.append("Voices: ");
    const chip = (text) => { const s = document.createElement("span"); s.className = "chip"; s.textContent = text; return s; };
    cast.append(chip("Narrator"));
    for (const c of Object.values(script.characters)) cast.append(chip(`${c.name}${c.voice ? " · " + c.voice : ""}`));
    root.append(cast);
    for (const ch of script.chapters) {
        const h = document.createElement("h4");
        h.textContent = ch.title;
        root.append(h);
        ch.scenes.forEach((sc, i) => {
            const div = document.createElement("div");
            div.className = "scene";
            div.append(`Scene ${i + 1} · ${sc.segments.length} lines · `);
            if (sc.ambience.length) sc.ambience.forEach((t) => div.append(chip(t))); else div.append("no ambience");
            root.append(div);
        });
    }
}

// ---------------------------------------------------------------- wiring

window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
    $("ambienceRange").addEventListener("input", () => { $("ambienceOut").textContent = `${$("ambienceRange").value} dB`; });
    $("ambienceCheck").addEventListener("change", () => { $("ambienceRange").disabled = !$("ambienceCheck").checked; });
    $("createBtn").addEventListener("click", createJob);

    loadEngines();
    loadHistory().catch(() => { /* server unreachable: leave the history empty */ });
});
