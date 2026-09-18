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

// 1. Initialize or Ensure Session via REST API
async function initSession() {
  try {
    const payload = {
      session_id: SESSION_ID,
      speaker_a: {
        speaker_id: "carlos",
        name: "Carlos",
        native_language: "es",
        preferred_voice_style: "default"
      },
      speaker_b: {
        speaker_id: "alice",
        name: "Alice",
        native_language: "en",
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

  teleprompterSocket = new WebSocket(wsUrl);

  teleprompterSocket.onopen = () => {
    statusDot.classList.add("connected");
    statusText.textContent = "En Vivo (Conectado)";
  };

  teleprompterSocket.onclose = () => {
    statusDot.classList.remove("connected");
    statusText.textContent = "Desconectado (Reintentando...)";
    setTimeout(connectTeleprompter, 2000);
  };

  teleprompterSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleIncomingEvent(data);
    } catch (e) {
      console.error("Failed to parse websocket message:", e);
    }
  };
}

// 3. Connect to Audio Stream WebSockets
function connectLocalMicStream() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio-stream/${SESSION_ID}/carlos`;

  localMicSocket = new WebSocket(wsUrl);

  localMicSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "turn_result" && data.audio_base64 && audioPlaybackToggle.checked) {
        playSynthesizedAudio(data.audio_base64, data.format || "wav");
      }
    } catch (e) {
      console.error("Local audio socket parse error:", e);
    }
  };
}

function connectMeetAudioStream() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio-stream/${SESSION_ID}/alice`;

  meetAudioSocket = new WebSocket(wsUrl);

  meetAudioSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "turn_result" && data.audio_base64 && audioPlaybackToggle.checked) {
        playSynthesizedAudio(data.audio_base64, data.format || "wav");
      }
    } catch (e) {
      console.error("Meet audio socket parse error:", e);
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

  if (payload.event_name === "TurnCompleted") {
    const turnData = payload.data;
    renderTurnCard(turnData);
  }
}

function renderTurnCard(turn) {
  const placeholder = feed.querySelector("p");
  if (placeholder) {
    feed.innerHTML = "";
  }

  const isSpeakerA = turn.speaker_id === "carlos" || turn.source_lang === "es";
  const speakerClass = isSpeakerA ? "speaker-a" : "speaker-b";
  const speakerName = isSpeakerA ? "Carlos (ES)" : "Alice (EN)";

  const card = document.createElement("div");
  card.className = `turn-card ${speakerClass}`;

  const timeStr = new Date().toLocaleTimeString();

  card.innerHTML = `
    <div class="turn-header">
      <span style="font-weight: 600;">${speakerName}</span>
      <span>${timeStr}</span>
    </div>
    <div class="turn-original">"${escapeHtml(turn.original_text)}"</div>
    <div class="turn-translated">➡️ ${escapeHtml(turn.translated_text)}</div>
  `;

  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
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

// 5. Microphone Recording (Push-to-talk for Carlos / Speaker A)
async function startLocalRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    localAudioChunks = [];
    localMicRecorder = new MediaRecorder(stream);

    localMicRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        localAudioChunks.push(e.data);
      }
    };

    localMicRecorder.onstop = async () => {
      const audioBlob = new Blob(localAudioChunks, { type: "audio/webm" });
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64Audio = btoa(
        new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), "")
      );

      if (localMicSocket && localMicSocket.readyState === WebSocket.OPEN) {
        localMicSocket.send(JSON.stringify({ audio_base64: base64Audio }));
      }
      stream.getTracks().forEach(track => track.stop());
    };

    localMicRecorder.start();
    isLocalRecording = true;
    recordBtn.classList.add("recording");
    recordText.textContent = "Escuchando... Soltá para Enviar";
  } catch (err) {
    alert("No se pudo acceder al micrófono: " + err.message);
  }
}

function stopLocalRecording() {
  if (localMicRecorder && isLocalRecording) {
    localMicRecorder.stop();
    isLocalRecording = false;
    recordBtn.classList.remove("recording");
    recordText.textContent = "Presionar para Hablar (ES)";
  }
}

recordBtn.addEventListener("mousedown", startLocalRecording);
recordBtn.addEventListener("mouseup", stopLocalRecording);
recordBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startLocalRecording(); });
recordBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopLocalRecording(); });

clearFeedBtn.addEventListener("click", () => {
  feed.innerHTML = `<div style="text-align: center; color: var(--text-muted); margin-top: 3rem;">
    <p>Teleprompter limpio.</p>
  </div>`;
});

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

// 7. Dual-Channel Capture for Remote Videocalls (Zoom / Google Meet)
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
        if (meetAudioSocket && meetAudioSocket.readyState === WebSocket.OPEN) {
          meetAudioSocket.send(JSON.stringify({ audio_base64: base64Audio }));
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
    meetAudioStatus.textContent = "🟢 Capturando Zoom/Meet";
    meetAudioStatus.style.color = "var(--success-color)";
    connectMeetAudioBtn.textContent = "🛑 Detener Captura";
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
    connectMeetAudioBtn.textContent = "📺 Capturar Audio de Reunión (Pestaña)";
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

// 8. FTS5 Historical Search Modal Logic
if (openSearchBtn && searchModal) {
  openSearchBtn.addEventListener("click", () => {
    searchModal.style.display = "flex";
    searchQueryInput.focus();
  });
}

if (closeSearchModalBtn && searchModal) {
  closeSearchModalBtn.addEventListener("click", () => {
    searchModal.style.display = "none";
  });
}

async function performSearch() {
  const q = searchQueryInput.value.trim();
  const topic = searchTopicInput.value.trim();
  if (!q) return;

  searchResultsList.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">Buscando en janus.db (SQLite FTS5)...</div>`;

  try {
    let url = `/api/meetings/search?q=${encodeURIComponent(q)}`;
    if (topic) {
      url += `&topic=${encodeURIComponent(topic)}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error("Error en la búsqueda");
    const results = await res.json();

    if (results.length === 0) {
      searchResultsList.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 2rem;">No se encontraron resultados para "${escapeHtml(q)}".</div>`;
      return;
    }

    searchResultsList.innerHTML = "";
    results.forEach(r => {
      const card = document.createElement("div");
      card.className = "search-hit-card";
      const topicTag = r.topic_key ? `<span style="background: rgba(0, 242, 254, 0.1); color: var(--accent-cyan); padding: 2px 6px; border-radius: 4px;">${escapeHtml(r.topic_key)}</span>` : "";
      const dateStr = new Date(r.created_at * 1000).toLocaleString();

      card.innerHTML = `
        <div class="search-hit-snippet">${r.snippet}</div>
        <div class="search-hit-meta">
          <span>Reunión: <strong>${escapeHtml(r.meeting_id)}</strong></span>
          <span>Hablante: ${escapeHtml(r.speaker_id)}</span>
          <span>${dateStr}</span>
          ${topicTag}
        </div>
      `;
      searchResultsList.appendChild(card);
    });
  } catch (err) {
    searchResultsList.innerHTML = `<div style="text-align: center; color: #ef4444; padding: 2rem;">Error: ${escapeHtml(err.message)}</div>`;
  }
}

if (executeSearchBtn) {
  executeSearchBtn.addEventListener("click", performSearch);
}
if (searchQueryInput) {
  searchQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") performSearch();
  });
}

// 9. Finalize Meeting & Zoom-Style Meeting Notes with AI Q&A
if (finalizeBtn) {
  finalizeBtn.addEventListener("click", async () => {
    finalizeBtn.disabled = true;
    finalizeBtn.textContent = "Generando Minuta...";

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
          li.style.display = "flex";
          li.style.alignItems = "center";
          li.style.gap = "0.6rem";
          const due = item.due_hint ? ` (Límite: ${item.due_hint})` : "";
          li.innerHTML = `<input type="checkbox" ${item.completed ? "checked" : ""} /> <span><strong>${escapeHtml(item.assignee)}:</strong> ${escapeHtml(item.task)}${due}</span>`;
          modalActionItems.appendChild(li);
        });

        notesModal.style.display = "flex";
      }
    } catch (err) {
      alert("Error generando minuta: " + err.message);
    } finally {
      finalizeBtn.disabled = false;
      finalizeBtn.textContent = "📋 Finalizar y Minuta (Zoom Style)";
    }
  });
}

if (closeModalBtn) {
  closeModalBtn.addEventListener("click", () => {
    notesModal.style.display = "none";
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
      copyNotesBtn.textContent = "✅ ¡Copiado!";
      setTimeout(() => { copyNotesBtn.textContent = "📋 Copiar Minuta"; }, 2000);
    } catch (err) {
      alert("No se pudo copiar: " + err.message);
    }
  });
}

// 10. BYOM: AI Chat with Meeting
if (modalAiProvider) {
  modalAiProvider.addEventListener("change", async () => {
    const provider = modalAiProvider.value;
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
  modalAiAskBtn.textContent = "Pensando...";
  modalAiAnswerBox.style.display = "block";
  modalAiAnswerText.textContent = "Consultando a la IA con el contexto de la reunión...";

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
    modalAiAnswerText.textContent = "⚠️ " + err.message;
  } finally {
    modalAiAskBtn.disabled = false;
    modalAiAskBtn.textContent = "Preguntar";
  }
}

if (modalAiAskBtn) {
  modalAiAskBtn.addEventListener("click", askAiAboutMeeting);
}
if (modalAiQuestionInput) {
  modalAiQuestionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") askAiAboutMeeting();
  });
}

// Bootstrap
window.addEventListener("DOMContentLoaded", async () => {
  await initSession();
  connectTeleprompter();
  connectLocalMicStream();
});
