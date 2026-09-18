// Janus Web Client & Teleprompter Controller

const SESSION_ID = "live_bilingual_room";
let teleprompterSocket = null;
let audioStreamSocket = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const feed = document.getElementById("feed");
const recordBtn = document.getElementById("recordBtn");
const recordText = document.getElementById("recordText");
const audioPlaybackToggle = document.getElementById("audioPlaybackToggle");
const clearFeedBtn = document.getElementById("clearFeedBtn");
const finalizeBtn = document.getElementById("finalizeBtn");
const notesModal = document.getElementById("notesModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalExecutiveSummary = document.getElementById("modalExecutiveSummary");
const modalKeyPoints = document.getElementById("modalKeyPoints");
const modalActionItems = document.getElementById("modalActionItems");
const downloadMdBtn = document.getElementById("downloadMdBtn");
const copyNotesBtn = document.getElementById("copyNotesBtn");

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

// 3. Connect to Audio Stream WebSocket
function connectAudioStream() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio-stream/${SESSION_ID}/carlos`;

  audioStreamSocket = new WebSocket(wsUrl);

  audioStreamSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "turn_result" && data.audio_base64 && audioPlaybackToggle.checked) {
        playSynthesizedAudio(data.audio_base64, data.format || "wav");
      }
    } catch (e) {
      console.error("Audio socket parse error:", e);
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

// 5. Microphone Recording Control (Push-to-talk)
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      const arrayBuffer = await audioBlob.arrayBuffer();
      const base64Audio = btoa(
        new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), "")
      );

      if (audioStreamSocket && audioStreamSocket.readyState === WebSocket.OPEN) {
        audioStreamSocket.send(JSON.stringify({ audio_base64: base64Audio }));
      }
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add("recording");
    recordText.textContent = "Escuchando... Soltá para Enviar";
  } catch (err) {
    alert("No se pudo acceder al micrófono: " + err.message);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordText.textContent = "Presionar para Hablar (ES)";
  }
}

recordBtn.addEventListener("mousedown", startRecording);
recordBtn.addEventListener("mouseup", stopRecording);
recordBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
recordBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });

clearFeedBtn.addEventListener("click", () => {
  feed.innerHTML = `<div style="text-align: center; color: var(--text-muted); margin-top: 3rem;">
    <p>Teleprompter limpio.</p>
  </div>`;
});

// 6. Finalize Meeting & Zoom-Style Meeting Notes
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

// Bootstrap
window.addEventListener("DOMContentLoaded", async () => {
  await initSession();
  connectTeleprompter();
  connectAudioStream();
});
