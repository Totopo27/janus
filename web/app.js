// Janus Web Client & Teleprompter Controller v0.2.0 (BYOM & Dual-Scenario Audio)

const SESSION_ID = "live_bilingual_room";
let teleprompterSocket = null;
let localMicSocket = null;
let meetAudioSocket = null;

let localMicRecorder = null;
let meetAudioRecorder = null;
let localAudioChunks = [];
let isLocalRecording = false;
let meetStream = null;

// DOM Elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const feed = document.getElementById("feed");
const recordBtn = document.getElementById("recordBtn");
const recordText = document.getElementById("recordText");
const audioPlaybackToggle = document.getElementById("audioPlaybackToggle");
const clearFeedBtn = document.getElementById("clearFeedBtn");
const finalizeBtn = document.getElementById("finalizeBtn");
const inputLanguageSelect = document.getElementById("inputLanguageSelect");

// Scenario Elements
const scenarioInPerson = document.getElementById("scenarioInPerson");
const scenarioVideocall = document.getElementById("scenarioVideocall");
const scenarioInPersonLabel = document.getElementById("scenarioInPersonLabel");
const scenarioVideocallLabel = document.getElementById("scenarioVideocallLabel");
const videocallControls = document.getElementById("videocallControls");
const connectMeetAudioBtn = document.getElementById("connectMeetAudioBtn");
const meetAudioStatus = document.getElementById("meetAudioStatus");

// Search Modal Elements
const openSearchBtn = document.getElementById("openSearchBtn");
const searchModal = document.getElementById("searchModal");
const closeSearchModalBtn = document.getElementById("closeSearchModalBtn");
const searchQueryInput = document.getElementById("searchQueryInput");
const searchTopicInput = document.getElementById("searchTopicInput");
const executeSearchBtn = document.getElementById("executeSearchBtn");
const searchResultsList = document.getElementById("searchResultsList");

// Notes & AI Chat Modal Elements
const notesModal = document.getElementById("notesModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalExecutiveSummary = document.getElementById("modalExecutiveSummary");
const modalKeyPoints = document.getElementById("modalKeyPoints");
const modalActionItems = document.getElementById("modalActionItems");
const downloadMdBtn = document.getElementById("downloadMdBtn");
const copyNotesBtn = document.getElementById("copyNotesBtn");
const modalAiProvider = document.getElementById("modalAiProvider");
const modalAiQuestionInput = document.getElementById("modalAiQuestionInput");
const modalAiAskBtn = document.getElementById("modalAiAskBtn");
const modalAiAnswerBox = document.getElementById("modalAiAnswerBox");
const modalAiAnswerText = document.getElementById("modalAiAnswerText");

// Janus Live Notetaker Elements
const liveTopicText = document.getElementById("liveTopicText");
const liveTakeawaysList = document.getElementById("liveTakeawaysList");
const liveActionItemsList = document.getElementById("liveActionItemsList");
const catchUpBtn = document.getElementById("catchUpBtn");
const catchUpBox = document.getElementById("catchUpBox");
const catchUpText = document.getElementById("catchUpText");
const closeCatchUpBtn = document.getElementById("closeCatchUpBtn");
const refreshLiveNotesBtn = document.getElementById("refreshLiveNotesBtn");


// 1. Initialize or Ensure Session via REST API
async function initSession() {
  try {
    const payload = {
      session_id: SESSION_ID,
      speaker_a: {
        speaker_id: "speaker_1",
        name: "Hablante 1",
        native_language: "es",
        preferred_voice_style: "default"
      },
      speaker_b: {
        speaker_id: "speaker_2",
        name: "Hablante 2",
        native_language: "es",
        preferred_voice_style: "default"
      }
    };

    await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    console.warn("Session may already exist or offline:", err);
  }
}

// 2. Connect to Teleprompter WebSocket
function connectTeleprompter() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live-notes/${SESSION_ID}`;

  console.log(`[TeleprompterSocket] Conectando a ${wsUrl}...`);
  teleprompterSocket = new WebSocket(wsUrl);

  teleprompterSocket.onopen = () => {
    console.log("[TeleprompterSocket] Conexión establecida (En Vivo)");
    statusDot.classList.add("connected");
    statusText.textContent = "En Vivo (Conectado)";
  };

  teleprompterSocket.onclose = (evt) => {
    console.warn(`[TeleprompterSocket] Conexión cerrada (código: ${evt.code}). Reintentando en 2s...`);
    statusDot.classList.remove("connected");
    statusText.textContent = "Desconectado (Reintentando...)";
    setTimeout(connectTeleprompter, 2000);
  };

  teleprompterSocket.onerror = (err) => {
    console.error("[TeleprompterSocket] Error en la conexión WebSocket:", err);
  };

  teleprompterSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("[TeleprompterSocket] Evento recibido:", data.event_name || data.type || data);
      handleIncomingEvent(data);
    } catch (e) {
      console.error("[TeleprompterSocket] Error parseando mensaje JSON:", e);
    }
  };
}

// 3. Connect to Audio Stream WebSockets
function connectLocalMicStream() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio-stream/${SESSION_ID}/local`;

  console.log(`[LocalMicSocket] Conectando a ${wsUrl}...`);
  localMicSocket = new WebSocket(wsUrl);

  localMicSocket.onopen = () => {
    console.log("[LocalMicSocket] Canal de audio local conectado exitosamente");
  };

  localMicSocket.onerror = (err) => {
    console.error("[LocalMicSocket] Error en el socket de audio local:", err);
  };

  localMicSocket.onclose = (evt) => {
    console.warn(`[LocalMicSocket] Canal de audio local cerrado (código: ${evt.code})`);
  };

  localMicSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("[LocalMicSocket] Resultado recibido del servidor:", data);
      // Play synthesized audio if enabled; rendering to teleprompter is handled by TurnCompleted event
      if (data.type === "turn_result" && data.audio_base64 && audioPlaybackToggle && audioPlaybackToggle.checked) {
        playSynthesizedAudio(data.audio_base64, data.format || "wav");
      }
    } catch (e) {
      console.error("[LocalMicSocket] Error parseando respuesta de audio:", e);
    }
  };
}

function connectMeetAudioStream() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio-stream/${SESSION_ID}/remote`;

  console.log(`[MeetAudioSocket] Conectando a ${wsUrl}...`);
  meetAudioSocket = new WebSocket(wsUrl);

  meetAudioSocket.onopen = () => {
    console.log("[MeetAudioSocket] Canal de audio remoto conectado exitosamente");
  };

  meetAudioSocket.onerror = (err) => {
    console.error("[MeetAudioSocket] Error en el socket de audio remoto:", err);
  };

  meetAudioSocket.onclose = (evt) => {
    console.warn(`[MeetAudioSocket] Canal de audio remoto cerrado (código: ${evt.code})`);
  };

  meetAudioSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("[MeetAudioSocket] Resultado recibido del servidor:", data);
      // Play synthesized audio if enabled; rendering to teleprompter is handled by TurnCompleted event
      if (data.type === "turn_result" && data.audio_base64 && audioPlaybackToggle && audioPlaybackToggle.checked) {
        playSynthesizedAudio(data.audio_base64, data.format || "wav");
      }
    } catch (e) {
      console.error("[MeetAudioSocket] Error parseando respuesta de audio remoto:", e);
    }
  };
}

// 4. Handle Incoming Domain Events for the Teleprompter
function handleIncomingEvent(payload) {
  if (payload.type === "history" && payload.turns && payload.turns.length > 0) {
    feed.innerHTML = "";
    payload.turns.forEach(renderTurnCard);
    return;
  }

  if (payload.event_name === "LiveNotesUpdated") {
    renderLiveNotes(payload.data || payload);
    return;
  }

  if (payload.event_name === "TurnCompleted") {
    const turnData = payload.data;
    renderTurnCard(turnData);
  }

}

const renderedTurnIds = new Set();

function renderTurnCard(turn) {
  if (!turn) return;

  const turnId = turn.turn_id || turn.id;
  if (turnId) {
    if (renderedTurnIds.has(turnId)) return;
    renderedTurnIds.add(turnId);
  }

  const placeholder = feed.querySelector(".feed-empty-state") || feed.querySelector("p");
  if (placeholder) {
    feed.innerHTML = "";
  }

  const srcLang = (turn.source_lang || turn.language || "es").toUpperCase();
  const tgtLang = (turn.target_lang || (srcLang === "ES" ? "EN" : "ES")).toUpperCase();

  let speakerClass = "speaker-a";
  let speakerLabel = turn.speaker_name || "Hablante 1";

  if (turn.speaker_id === "speaker_2" || turn.speaker_id === "remote") {
    speakerClass = "speaker-b";
    speakerLabel = turn.speaker_name || "Hablante 2";
  } else if (turn.speaker_id === "speaker_1" || turn.speaker_id === "local") {
    speakerClass = "speaker-a";
    speakerLabel = turn.speaker_name || "Hablante 1";
  }

  const speakerHeader = `${speakerLabel} (${srcLang} → ${tgtLang})`;

  const card = document.createElement("article");
  card.className = `turn-card ${speakerClass}`;
  if (turnId) card.id = `turn-${turnId}`;
  card.setAttribute("aria-label", `Turno de ${speakerHeader}`);

  const timeStr = new Date().toLocaleTimeString();
  const arrowSvg = `<svg class="translation-indicator" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>`;

  card.innerHTML = `
    <div class="turn-header">
      <span style="font-weight: 600;">${speakerHeader}</span>
      <time datetime="${new Date().toISOString()}">${timeStr}</time>
    </div>
    <div class="turn-original">“${escapeHtml(turn.original_text)}”</div>
    <div class="turn-translated">${arrowSvg} <span>${escapeHtml(turn.translated_text)}</span></div>
  `;

  feed.appendChild(card);
  feed.scrollTo({ top: feed.scrollHeight, behavior: "smooth" });
}

function playSynthesizedAudio(base64Data, format) {
  const audio = new Audio(`data:audio/${format};base64,${base64Data}`);
  audio.play().catch(e => console.warn("Auto-play prevented or failed:", e));
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Universal Audio Channel Selection
let currentSpeakerId = "local";

// 5. Microphone Recording (Toggle On/Off)
async function startLocalRecording() {
  try {
    if (!localMicSocket || localMicSocket.readyState !== WebSocket.OPEN) {
      console.warn("[AudioRecorder] Socket de audio local no está conectado. Reintentando...");
      connectLocalMicStream();
    }

    console.log("[AudioRecorder] Solicitando acceso al micrófono...");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    localAudioChunks = [];
    localMicRecorder = new MediaRecorder(stream);

    localMicRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        localAudioChunks.push(e.data);
        console.log(`[AudioRecorder] Bloque de audio grabado: ${e.data.size} bytes`);
      }
    };

    localMicRecorder.onstop = async () => {
      const audioBlob = new Blob(localAudioChunks, { type: "audio/webm" });
      console.log(`[AudioRecorder] Grabación detenida. Total bloques: ${localAudioChunks.length}, tamaño blob: ${audioBlob.size} bytes`);

      if (audioBlob.size === 0) {
        console.warn("[AudioRecorder] El blob de audio resultó vacío (0 bytes). Se omite el envío.");
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64Audio = btoa(
        new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), "")
      );

      const selectedLang = inputLanguageSelect ? inputLanguageSelect.value : "es";
      console.log(`[AudioRecorder] Enviando audio base64 (${base64Audio.length} caracteres, idioma: ${selectedLang}) al servidor...`);

      if (localMicSocket && localMicSocket.readyState === WebSocket.OPEN) {
        localMicSocket.send(JSON.stringify({
          audio_base64: base64Audio,
          mime_type: localMicRecorder.mimeType || "audio/webm",
          language: selectedLang,
        }));
        console.log(`[AudioRecorder] Audio enviado exitosamente por WebSocket (idioma: ${selectedLang})`);
      } else {
        console.error("[AudioRecorder] No se pudo enviar el audio: WebSocket cerrado o no listo");
      }

      stream.getTracks().forEach(track => track.stop());
    };

    localMicRecorder.start();
    isLocalRecording = true;
    recordBtn.classList.add("recording");
    if (recordText) recordText.textContent = "Detener Grabación";
    recordBtn.setAttribute("aria-label", "Detener grabación de audio");
    console.log("[AudioRecorder] Grabación iniciada en modo Universal");
  } catch (err) {
    console.error("[AudioRecorder] Error accediendo al micrófono:", err);
    alert("No se pudo acceder al micrófono: " + err.message);
  }
}

function stopLocalRecording() {
  if (localMicRecorder && isLocalRecording) {
    console.log("[AudioRecorder] Deteniendo grabación...");
    localMicRecorder.stop();
    isLocalRecording = false;
    recordBtn.classList.remove("recording");
    if (recordText) recordText.textContent = "Iniciar Grabación";
    recordBtn.setAttribute("aria-label", "Iniciar grabación de audio");
  }
}

// Toggle recording on button click
if (recordBtn) {
  recordBtn.addEventListener("click", () => {
    if (isLocalRecording) {
      stopLocalRecording();
    } else {
      startLocalRecording();
    }
  });
}

if (clearFeedBtn) {
  clearFeedBtn.addEventListener("click", () => {
    renderedTurnIds.clear();
    feed.innerHTML = `
      <div class="feed-empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="empty-icon" aria-hidden="true" focusable="false">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/>
        </svg>
        <p>Presioná el botón de voz para iniciar la captura.</p>
        <span>Las transcripciones originales y traducciones en vivo aparecerán aquí con tipografía optimizada.</span>
      </div>
    `;
  });
}

// 6. Scenario Selector Handling (In-person vs Videocall Dual-Channel)
if (scenarioInPerson && scenarioVideocall) {
  scenarioInPerson.addEventListener("change", () => {
    if (scenarioInPerson.checked) {
      scenarioInPersonLabel.classList.add("active");
      scenarioVideocallLabel.classList.remove("active");
      videocallControls.style.display = "none";
      stopMeetAudioCapture();
    }
  });

  scenarioVideocall.addEventListener("change", () => {
    if (scenarioVideocall.checked) {
      scenarioVideocallLabel.classList.add("active");
      scenarioInPersonLabel.classList.remove("active");
      videocallControls.style.display = "flex";
      connectMeetAudioStream();
    }
  });
}

// 7. Dual-Channel Capture for Remote Videocalls
async function startMeetAudioCapture() {
  try {
    meetStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: true
    });

    const audioTracks = meetStream.getAudioTracks();
    if (audioTracks.length === 0) {
      alert("Asegurate de marcar la casilla 'Compartir audio de la pestaña/sistema' al seleccionar la ventana.");
      stopMeetAudioCapture();
      return;
    }

    const audioOnlyStream = new MediaStream([audioTracks[0]]);
    meetAudioRecorder = new MediaRecorder(audioOnlyStream);

    let chunks = [];
    meetAudioRecorder.ondataavailable = async (e) => {
      if (e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    // Slice audio every 2.5s for real-time translation loop
    const intervalId = setInterval(async () => {
      if (meetAudioRecorder && meetAudioRecorder.state === "recording") {
        meetAudioRecorder.stop();
        meetAudioRecorder.start();
      }
    }, 2500);

    meetAudioRecorder.onstop = async () => {
      if (chunks.length > 0) {
        const audioBlob = new Blob(chunks, { type: "audio/webm" });
        chunks = [];
        const arrayBuffer = await audioBlob.arrayBuffer();
        const base64Audio = btoa(
          new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), "")
        );
        const selectedLang = inputLanguageSelect ? inputLanguageSelect.value : "auto";
        if (meetAudioSocket && meetAudioSocket.readyState === WebSocket.OPEN) {
          meetAudioSocket.send(JSON.stringify({
            audio_base64: base64Audio,
            language: selectedLang,
          }));
        }
      }
    };

    meetStream.getVideoTracks().forEach(track => {
      track.onended = () => {
        clearInterval(intervalId);
        stopMeetAudioCapture();
      };
    });

    meetAudioRecorder.start();
    meetAudioStatus.textContent = "Capturando audio remoto";
    meetAudioStatus.style.color = "var(--accent-emerald)";
    connectMeetAudioBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 9h6v6H9z"/></svg>
      <span>Detener Captura</span>
    `;
  } catch (err) {
    console.error("Error capturing meet audio:", err);
    meetAudioStatus.textContent = "Error de captura";
  }
}

function stopMeetAudioCapture() {
  if (meetAudioRecorder && meetAudioRecorder.state !== "inactive") {
    meetAudioRecorder.stop();
  }
  if (meetStream) {
    meetStream.getTracks().forEach(t => t.stop());
    meetStream = null;
  }
  meetAudioStatus.textContent = "Inactivo";
  meetAudioStatus.style.color = "var(--text-muted)";
  if (connectMeetAudioBtn) {
    connectMeetAudioBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Capturar Audio Remoto (Pestaña)</span>
    `;
  }
}

if (connectMeetAudioBtn) {
  connectMeetAudioBtn.addEventListener("click", () => {
    if (meetStream) {
      stopMeetAudioCapture();
    } else {
      startMeetAudioCapture();
    }
  });
}

// 8. FTS5 Historical Search Modal Logic & Safe Snippet Sanitization
function sanitizeFtsSnippet(rawSnippet) {
  if (!rawSnippet) return "";
  // Escapar HTML malicioso preservando únicamente los tags <mark> y </mark> de SQLite FTS5
  const escaped = escapeHtml(rawSnippet);
  return escaped.replace(/&lt;mark&gt;/g, "<mark>").replace(/&lt;\/mark&gt;/g, "</mark>");
}

if (openSearchBtn && searchModal) {
  openSearchBtn.addEventListener("click", () => {
    searchModal.style.display = "flex";
    if (searchQueryInput) searchQueryInput.focus();
  });
}

if (closeSearchModalBtn && searchModal) {
  closeSearchModalBtn.addEventListener("click", () => {
    searchModal.style.display = "none";
    if (openSearchBtn) openSearchBtn.focus();
  });
}

async function performSearch() {
  const q = searchQueryInput.value.trim();
  const topic = searchTopicInput.value.trim();
  if (!q) return;

  searchResultsList.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">Buscando en janus.db (SQLite FTS5)…</div>`;

  try {
    let url = `/api/meetings/search?q=${encodeURIComponent(q)}`;
    if (topic) {
      url += `&topic=${encodeURIComponent(topic)}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error("Error en la búsqueda");
    const results = await res.json();

    if (results.length === 0) {
      searchResultsList.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">No se encontraron resultados para “${escapeHtml(q)}”.</div>`;
      return;
    }

    searchResultsList.innerHTML = "";
    results.forEach(r => {
      const card = document.createElement("article");
      card.className = "search-hit-card";
      const topicTag = r.topic_key ? `<span style="background: rgba(0, 210, 160, 0.12); color: var(--accent-telemetry); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.7rem;">${escapeHtml(r.topic_key)}</span>` : "";
      const dateStr = new Date(r.created_at * 1000).toLocaleString();

      card.innerHTML = `
        <div class="search-hit-snippet">${sanitizeFtsSnippet(r.snippet)}</div>
        <div class="search-hit-meta">
          <span>Reunión: <strong>${escapeHtml(r.meeting_id)}</strong></span>
          <span>Hablante: ${escapeHtml(r.speaker_id)}</span>
          <time datetime="${new Date(r.created_at * 1000).toISOString()}">${dateStr}</time>
          ${topicTag}
        </div>
      `;
      searchResultsList.appendChild(card);
    });
  } catch (err) {
    searchResultsList.innerHTML = `<div style="text-align: center; color: var(--accent-live); padding: 2rem;">Error: ${escapeHtml(err.message)}</div>`;
  }
}

if (executeSearchBtn) {
  executeSearchBtn.addEventListener("click", performSearch);
}
if (searchQueryInput) {
  searchQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      performSearch();
    }
  });
}
const searchForm = document.getElementById("searchForm");
if (searchForm) {
  searchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    performSearch();
  });
}

// 9. Finalize Meeting & Meeting Notes with AI Q&A
if (finalizeBtn) {
  finalizeBtn.addEventListener("click", async () => {
    finalizeBtn.disabled = true;
    finalizeBtn.innerHTML = `<span>Generando Minuta…</span>`;

    try {
      const res = await fetch(`/api/meetings/${SESSION_ID}/finalize`, { method: "POST" });
      if (!res.ok) {
        throw new Error("No se pudo finalizar la reunión");
      }
      const data = await res.json();
      const summary = data.summary;

      if (summary) {
        modalExecutiveSummary.textContent = summary.executive_summary;

        modalKeyPoints.innerHTML = "";
        (summary.key_points || []).forEach(pt => {
          const li = document.createElement("li");
          li.textContent = pt;
          modalKeyPoints.appendChild(li);
        });

        modalActionItems.innerHTML = "";
        (summary.action_items || []).forEach(item => {
          const li = document.createElement("li");
          const due = item.due_hint ? ` (Límite: ${item.due_hint})` : "";
          li.innerHTML = `<input type="checkbox" id="action_${Math.random().toString(36).substring(2, 7)}" ${item.completed ? "checked" : ""} aria-label="Estado de tarea: ${escapeHtml(item.task)}" /> <span><strong>${escapeHtml(item.assignee)}:</strong> ${escapeHtml(item.task)}${due}</span>`;
          modalActionItems.appendChild(li);
        });

        notesModal.style.display = "flex";
        if (closeModalBtn) closeModalBtn.focus();
      }
    } catch (err) {
      alert("Error generando minuta: " + err.message);
    } finally {
      finalizeBtn.disabled = false;
      finalizeBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
        <span>Generar Minuta</span>
      `;
    }
  });
}

if (closeModalBtn) {
  closeModalBtn.addEventListener("click", () => {
    notesModal.style.display = "none";
    if (finalizeBtn) finalizeBtn.focus();
  });
}

if (downloadMdBtn) {
  downloadMdBtn.addEventListener("click", () => {
    window.location.href = `/api/meetings/${SESSION_ID}/notes?format=markdown`;
  });
}

if (copyNotesBtn) {
  copyNotesBtn.addEventListener("click", async () => {
    try {
      const res = await fetch(`/api/meetings/${SESSION_ID}/notes?format=markdown`);
      const md = await res.text();
      await navigator.clipboard.writeText(md);
      copyNotesBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><polyline points="20 6 9 17 4 12"/></svg>
        <span>Copiado al Portapapeles</span>
      `;
      setTimeout(() => {
        copyNotesBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
          <span>Copiar Minuta</span>
        `;
      }, 2000);
    } catch (err) {
      alert("No se pudo copiar: " + err.message);
    }
  });
}

// 10. BYOM: AI Chat with Meeting & Inference Badge Sync
const byomEngineName = document.getElementById("byomEngineName");

function updateByomBadge(provider) {
  if (!byomEngineName) return;
  if (provider === "gemini") {
    byomEngineName.textContent = "Google Gemini (Cloud)";
  } else {
    byomEngineName.textContent = "Ollama Local (qwen2.5:3b)";
  }
}

if (modalAiProvider) {
  updateByomBadge(modalAiProvider.value);
  modalAiProvider.addEventListener("change", async () => {
    const provider = modalAiProvider.value;
    updateByomBadge(provider);
    try {
      await fetch("/api/system/llm-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider })
      });
    } catch (e) {
      console.warn("Could not switch LLM provider:", e);
    }
  });
}

async function askAiAboutMeeting() {
  const question = modalAiQuestionInput.value.trim();
  if (!question) return;

  modalAiAskBtn.disabled = true;
  modalAiAskBtn.textContent = "Pensando…";
  modalAiAnswerBox.style.display = "block";
  modalAiAnswerText.textContent = "Consultando a la IA con el contexto de la reunión…";

  try {
    const res = await fetch(`/api/meetings/${SESSION_ID}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Error al consultar la IA");
    }
    const data = await res.json();
    modalAiAnswerText.innerHTML = `<strong>Respuesta (${escapeHtml(data.provider)}):</strong><br/>${escapeHtml(data.answer).replace(/\n/g, "<br/>")}`;
  } catch (err) {
    modalAiAnswerText.textContent = "Error: " + err.message;
  } finally {
    modalAiAskBtn.disabled = false;
    modalAiAskBtn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      <span>Consultar</span>
    `;
  }
}

if (modalAiAskBtn) {
  modalAiAskBtn.addEventListener("click", askAiAboutMeeting);
}
if (modalAiQuestionInput) {
  modalAiQuestionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      askAiAboutMeeting();
    }
  });
}
const byomForm = document.getElementById("byomForm");
if (byomForm) {
  byomForm.addEventListener("submit", (e) => {
    e.preventDefault();
    askAiAboutMeeting();
  });
}

// 11. Janus Live Notes & Catch Me Up Controller
function renderLiveNotes(data) {
  if (!data) return;
  if (data.current_topic && liveTopicText) {
    liveTopicText.textContent = data.current_topic;
  }

  if (data.key_takeaways && liveTakeawaysList) {
    if (data.key_takeaways.length === 0) {
      liveTakeawaysList.innerHTML = `<li class="empty-hint">El asistente está escuchando activamente para sintetizar los acuerdos…</li>`;
    } else {
      liveTakeawaysList.innerHTML = "";
      data.key_takeaways.forEach(pt => {
        const li = document.createElement("li");
        li.textContent = pt;
        liveTakeawaysList.appendChild(li);
      });
    }
  }

  if (data.action_items && liveActionItemsList) {
    if (data.action_items.length === 0) {
      liveActionItemsList.innerHTML = `<li class="empty-hint">Aún no se han detectado compromisos o tareas explícitas.</li>`;
    } else {
      liveActionItemsList.innerHTML = "";
      data.action_items.forEach(itm => {
        const li = document.createElement("li");
        const due = itm.due_hint ? ` (Plazo: ${escapeHtml(itm.due_hint)})` : "";
        li.innerHTML = `<input type="checkbox" id="live_action_${Math.random().toString(36).substring(2, 7)}" ${itm.completed ? "checked" : ""} aria-label="Tarea para ${escapeHtml(itm.assignee)}: ${escapeHtml(itm.task)}" /> <span><strong>${escapeHtml(itm.assignee)}:</strong> ${escapeHtml(itm.task)}${due}</span>`;
        liveActionItemsList.appendChild(li);
      });
    }
  }
}

async function loadInitialLiveNotes() {
  try {
    const res = await fetch(`/api/meetings/${SESSION_ID}/live-notes`);
    if (res.ok) {
      const data = await res.json();
      renderLiveNotes(data);
    }
  } catch (e) {
    console.debug("Live notes initial load skipped:", e);
  }
}

if (catchUpBtn) {
  catchUpBtn.addEventListener("click", async () => {
    catchUpBtn.disabled = true;
    catchUpBtn.innerHTML = `<span>Sintetizando…</span>`;
    catchUpBox.style.display = "block";
    catchUpText.textContent = "El asistente Janus está revisando los últimos minutos de la conversación…";

    try {
      const res = await fetch(`/api/meetings/${SESSION_ID}/catch-up`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ last_n_turns: 6 }),
      });
      if (!res.ok) throw new Error("No se pudo obtener el resumen");
      const data = await res.json();
      catchUpText.innerHTML = escapeHtml(data.summary).replace(/\n/g, "<br/>");
    } catch (err) {
      catchUpText.textContent = "Error: " + err.message;
    } finally {
      catchUpBtn.disabled = false;
      catchUpBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        <span>Ponerse al Día</span>
      `;
    }
  });
}

if (closeCatchUpBtn) {
  closeCatchUpBtn.addEventListener("click", () => {
    catchUpBox.style.display = "none";
    if (catchUpBtn) catchUpBtn.focus();
  });
}

if (refreshLiveNotesBtn) {
  refreshLiveNotesBtn.addEventListener("click", async () => {
    refreshLiveNotesBtn.style.transform = "rotate(180deg)";
    try {
      const res = await fetch(`/api/meetings/${SESSION_ID}/live-notes/refresh`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        renderLiveNotes(data);
      }
    } catch (e) {
      console.warn("Could not refresh live notes:", e);
    } finally {
      setTimeout(() => { refreshLiveNotesBtn.style.transform = "none"; }, 500);
    }
  });
}

// Global Keyboard Shortcuts (Escape to close modals/drawers) & Backdrop Click
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (searchModal && searchModal.style.display === "flex") {
      searchModal.style.display = "none";
      if (openSearchBtn) openSearchBtn.focus();
    }
    if (notesModal && notesModal.style.display === "flex") {
      notesModal.style.display = "none";
      if (finalizeBtn) finalizeBtn.focus();
    }
    if (catchUpBox && catchUpBox.style.display === "block") {
      catchUpBox.style.display = "none";
      if (catchUpBtn) catchUpBtn.focus();
    }
  }
});

// Click outside modal content to dismiss
[searchModal, notesModal].forEach(modal => {
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.style.display = "none";
      }
    });
  }
});

// Bootstrap
window.addEventListener("DOMContentLoaded", async () => {
  await initSession();
  connectTeleprompter();
  connectLocalMicStream();
  await loadInitialLiveNotes();
});

