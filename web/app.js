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
const vadModeToggle = document.getElementById("vadModeToggle");
const audioPlaybackToggle = document.getElementById("audioPlaybackToggle");
const clearFeedBtn = document.getElementById("clearFeedBtn");
const finalizeBtn = document.getElementById("finalizeBtn");
const inputLanguageSelect = document.getElementById("inputLanguageSelect");
const vadTelemetryBar = document.getElementById("vadTelemetryBar");
const vadStatusDot = document.getElementById("vadStatusDot");
const vadStatusText = document.getElementById("vadStatusText");
const vadVolumeBar = document.getElementById("vadVolumeBar");
const vadBars = document.querySelectorAll(".vad-bar");

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
  } else {
    const num = parseInt(String(turn.speaker_id).replace("speaker_", ""), 10);
    speakerClass = (!isNaN(num) && num % 2 === 0) ? "speaker-b" : "speaker-a";
    speakerLabel = turn.speaker_name || `Hablante ${num || 1}`;
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

// 5. Dual-Mode Recording Architecture: Manual Studio Mode vs Hands-Free VAD Mode
let vadAudioContext = null;
let vadAnalyser = null;
let vadSource = null;
let vadStream = null;
let vadAnimationId = null;
let vadCurrentRecorder = null;
let vadCurrentChunks = [];
let vadState = "IDLE"; // "IDLE" | "RECORDING_MANUAL" | "LISTENING" | "SPEAKING" | "DISPATCHING"
let vadSpeechStartTime = 0;
let vadLastSpeechTime = 0;

const VAD_CONFIG = {
  speechStartThreshold: 0.022,    // RMS to transition from LISTENING to SPEAKING
  speechContinueThreshold: 0.015, // Hysteresis threshold to maintain SPEAKING
  minSpeechDurationMs: 600,       // Minimum duration of voice for a valid turn (avoids throat clears/clicks)
  silencePauseMs: 1400,           // 1.4s natural pause before completing turn (prevents mid-sentence cuts)
  maxTurnDurationMs: 16000,       // Max speech duration before forced turn slice
};

function getSupportedMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const t of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return "";
}

function updateVadUI(state, rms = 0) {
  if (!vadTelemetryBar) return;

  vadTelemetryBar.classList.remove("listening", "speaking", "dispatching");

  if (state === "IDLE") {
    vadTelemetryBar.classList.add("hidden");
    if (vadVolumeBar) vadVolumeBar.style.width = "0%";
    if (vadBars) vadBars.forEach(b => b.style.height = "20%");
  } else if (state === "RECORDING_MANUAL") {
    vadTelemetryBar.classList.remove("hidden");
    vadTelemetryBar.classList.add("speaking");
    if (vadStatusText) vadStatusText.textContent = "Grabando turno... (Presioná 'Detener' al finalizar)";
  } else if (state === "LISTENING") {
    vadTelemetryBar.classList.remove("hidden");
    vadTelemetryBar.classList.add("listening");
    if (vadStatusText) vadStatusText.textContent = "Escuchando... (Manos libres activo)";
  } else if (state === "SPEAKING") {
    vadTelemetryBar.classList.remove("hidden");
    vadTelemetryBar.classList.add("speaking");
    if (vadStatusText) vadStatusText.textContent = "Detectando voz...";
  } else if (state === "DISPATCHING") {
    vadTelemetryBar.classList.remove("hidden");
    vadTelemetryBar.classList.add("dispatching");
    if (vadStatusText) vadStatusText.textContent = "Procesando turno con Whisper Small...";
  }
}

function updateVadMeterUI(rms) {
  if (!vadVolumeBar && (!vadBars || vadBars.length === 0)) return;

  const pct = Math.min(100, Math.round((rms / 0.12) * 100));

  if (vadVolumeBar) {
    vadVolumeBar.style.width = `${pct}%`;
  }

  if (vadBars && vadBars.length > 0) {
    const heights = [
      Math.max(15, Math.min(100, pct * 0.7)),
      Math.max(20, Math.min(100, pct * 1.1)),
      Math.max(25, Math.min(100, pct * 1.4)),
      Math.max(20, Math.min(100, pct * 1.0)),
      Math.max(15, Math.min(100, pct * 0.6)),
    ];
    vadBars.forEach((bar, idx) => {
      bar.style.height = `${heights[idx]}%`;
    });
  }
}

function startNewRecorderSlice() {
  if (!vadStream || !vadStream.active) return;

  const mimeType = getSupportedMimeType();
  vadCurrentChunks = [];
  const options = mimeType ? { mimeType } : {};
  vadCurrentRecorder = new MediaRecorder(vadStream, options);

  vadCurrentRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      vadCurrentChunks.push(e.data);
    }
  };

  vadCurrentRecorder.start(250);
}

function sliceAndDispatchTurn() {
  if (!vadCurrentRecorder || vadCurrentRecorder.state !== "recording") {
    return;
  }

  const completedRecorder = vadCurrentRecorder;
  const completedChunks = vadCurrentChunks;

  // Immediately start next slice so no voice frames are lost
  startNewRecorderSlice();

  completedRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      completedChunks.push(e.data);
    }
  };

  completedRecorder.onstop = async () => {
    if (completedChunks.length === 0) return;
    const blob = new Blob(completedChunks, { type: completedRecorder.mimeType || "audio/webm" });
    if (blob.size < 800) {
      console.log(`[HandsFreeVAD] Fragmento muy corto descartado (${blob.size} bytes)`);
      return;
    }
    await sendAudioBlobToServer(blob, completedRecorder.mimeType || "audio/webm");
  };

  completedRecorder.stop();
}

async function sendAudioBlobToServer(blob, mimeType) {
  try {
    if (!localMicSocket || localMicSocket.readyState !== WebSocket.OPEN) {
      console.warn("[AudioCapture] WebSocket no conectado, reconectando...");
      connectLocalMicStream();
      await new Promise(r => setTimeout(r, 250));
    }

    const arrayBuffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    let binary = "";
    const chunkSize = 8192;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
    }
    const base64Audio = btoa(binary);
    const selectedLang = inputLanguageSelect ? inputLanguageSelect.value : "auto";

    console.log(`[AudioCapture] Despachando turno (${blob.size} bytes, lang=${selectedLang})...`);
    localMicSocket.send(JSON.stringify({
      audio_base64: base64Audio,
      mime_type: mimeType,
      language: selectedLang,
    }));
  } catch (err) {
    console.error("[AudioCapture] Error enviando audio al servidor:", err);
  }
}

function runVadLoop() {
  if (!isLocalRecording || !vadAnalyser) return;

  const bufferLength = vadAnalyser.fftSize;
  const dataArray = new Float32Array(bufferLength);

  const loop = () => {
    if (!isLocalRecording || !vadAnalyser) return;

    vadAnalyser.getFloatTimeDomainData(dataArray);

    let sumSquares = 0;
    for (let i = 0; i < bufferLength; i++) {
      sumSquares += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sumSquares / bufferLength);
    const now = Date.now();

    updateVadMeterUI(rms);

    if (vadState === "LISTENING") {
      if (rms >= VAD_CONFIG.speechStartThreshold) {
        vadState = "SPEAKING";
        vadSpeechStartTime = now;
        vadLastSpeechTime = now;
        updateVadUI("SPEAKING", rms);
      }
    } else if (vadState === "SPEAKING") {
      if (rms >= VAD_CONFIG.speechContinueThreshold) {
        vadLastSpeechTime = now;
      }

      const silenceDuration = now - vadLastSpeechTime;
      const speechDuration = vadLastSpeechTime - vadSpeechStartTime;
      const totalTurnDuration = now - vadSpeechStartTime;

      // Natural pause detected after speech (1.4s threshold)
      if (silenceDuration >= VAD_CONFIG.silencePauseMs) {
        if (speechDuration >= VAD_CONFIG.minSpeechDurationMs) {
          console.log(`[HandsFreeVAD] Pausa detectada (${silenceDuration}ms). Turno completado (${speechDuration}ms).`);
          updateVadUI("DISPATCHING", 0);
          sliceAndDispatchTurn();
        } else {
          console.log(`[HandsFreeVAD] Descartado ruido breve (${speechDuration}ms).`);
        }
        vadState = "LISTENING";
        setTimeout(() => {
          if (isLocalRecording && vadState === "LISTENING") {
            updateVadUI("LISTENING", 0);
          }
        }, 500);
      } else if (totalTurnDuration >= VAD_CONFIG.maxTurnDurationMs) {
        console.log(`[HandsFreeVAD] Límite de duración continua alcanzado (${totalTurnDuration}ms). Despachando turno.`);
        updateVadUI("DISPATCHING", 0);
        sliceAndDispatchTurn();
        vadSpeechStartTime = now;
        vadLastSpeechTime = now;
      }
    }

    vadAnimationId = requestAnimationFrame(loop);
  };

  vadAnimationId = requestAnimationFrame(loop);
}

function runManualMeterLoop() {
  if (!isLocalRecording || !vadAnalyser) return;

  const bufferLength = vadAnalyser.fftSize;
  const dataArray = new Float32Array(bufferLength);

  const loop = () => {
    if (!isLocalRecording || !vadAnalyser) return;

    vadAnalyser.getFloatTimeDomainData(dataArray);

    let sumSquares = 0;
    for (let i = 0; i < bufferLength; i++) {
      sumSquares += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sumSquares / bufferLength);

    updateVadMeterUI(rms);

    vadAnimationId = requestAnimationFrame(loop);
  };

  vadAnimationId = requestAnimationFrame(loop);
}

async function startLocalRecording() {
  try {
    if (!localMicSocket || localMicSocket.readyState !== WebSocket.OPEN) {
      console.warn("[AudioCapture] Socket de audio local desconectado. Conectando...");
      connectLocalMicStream();
    }

    console.log("[AudioCapture] Solicitando acceso al micrófono...");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });

    vadStream = stream;
    vadAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    vadSource = vadAudioContext.createMediaStreamSource(stream);
    vadAnalyser = vadAudioContext.createAnalyser();
    vadAnalyser.fftSize = 512;
    vadAnalyser.smoothingTimeConstant = 0.3;
    vadSource.connect(vadAnalyser);

    isLocalRecording = true;
    recordBtn.classList.add("recording");
    if (recordText) recordText.textContent = "Detener Grabación";
    recordBtn.setAttribute("aria-label", "Detener captura de audio");

    const isHandsFree = vadModeToggle && vadModeToggle.checked;

    if (isHandsFree) {
      vadState = "LISTENING";
      updateVadUI("LISTENING", 0);
      startNewRecorderSlice();
      runVadLoop();
      console.log("[AudioCapture] Captura Manos Libres (VAD automático) iniciada.");
    } else {
      vadState = "RECORDING_MANUAL";
      updateVadUI("RECORDING_MANUAL", 0);
      const mimeType = getSupportedMimeType();
      vadCurrentChunks = [];
      const options = mimeType ? { mimeType } : {};
      vadCurrentRecorder = new MediaRecorder(vadStream, options);
      vadCurrentRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          vadCurrentChunks.push(e.data);
        }
      };
      vadCurrentRecorder.start(250);
      runManualMeterLoop();
      console.log("[AudioCapture] Grabación Manual de Turno iniciada (sin cortes automáticos).");
    }
  } catch (err) {
    console.error("[AudioCapture] Error accediendo al micrófono:", err);
    alert("No se pudo acceder al micrófono: " + err.message);
  }
}

function stopLocalRecording() {
  if (!isLocalRecording) return;
  console.log("[AudioCapture] Deteniendo grabación...");
  isLocalRecording = false;

  if (vadAnimationId) {
    cancelAnimationFrame(vadAnimationId);
    vadAnimationId = null;
  }

  const isHandsFree = vadModeToggle && vadModeToggle.checked;
  const now = Date.now();

  if (isHandsFree) {
    // If there was ongoing speech when stopped, dispatch it
    if (vadState === "SPEAKING" && (now - vadSpeechStartTime >= VAD_CONFIG.minSpeechDurationMs)) {
      if (vadCurrentRecorder && vadCurrentRecorder.state === "recording") {
        const finalRecorder = vadCurrentRecorder;
        const finalChunks = vadCurrentChunks;
        finalRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) finalChunks.push(e.data);
        };
        finalRecorder.onstop = async () => {
          if (finalChunks.length > 0) {
            const blob = new Blob(finalChunks, { type: finalRecorder.mimeType || "audio/webm" });
            if (blob.size >= 800) {
              await sendAudioBlobToServer(blob, finalRecorder.mimeType || "audio/webm");
            }
          }
        };
        finalRecorder.stop();
      }
    } else if (vadCurrentRecorder && vadCurrentRecorder.state === "recording") {
      vadCurrentRecorder.stop();
    }
  } else {
    // Manual Mode: Dispatch the complete, uninterrupted audio turn
    if (vadCurrentRecorder && vadCurrentRecorder.state === "recording") {
      updateVadUI("DISPATCHING", 0);
      const finalRecorder = vadCurrentRecorder;
      const finalChunks = vadCurrentChunks;
      finalRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) finalChunks.push(e.data);
      };
      finalRecorder.onstop = async () => {
        if (finalChunks.length > 0) {
          const blob = new Blob(finalChunks, { type: finalRecorder.mimeType || "audio/webm" });
          if (blob.size >= 800) {
            await sendAudioBlobToServer(blob, finalRecorder.mimeType || "audio/webm");
          }
        }
      };
      finalRecorder.stop();
    }
  }

  vadState = "IDLE";
  vadCurrentRecorder = null;
  vadCurrentChunks = [];

  if (vadStream) {
    vadStream.getTracks().forEach((track) => track.stop());
    vadStream = null;
  }

  if (vadAudioContext && vadAudioContext.state !== "closed") {
    vadAudioContext.close().catch(() => {});
    vadAudioContext = null;
  }

  recordBtn.classList.remove("recording");
  if (recordText) {
    recordText.textContent = isHandsFree ? "Iniciar Grabación (Manos Libres)" : "Grabar Turno (Manual)";
  }
  recordBtn.setAttribute("aria-label", "Iniciar grabación de audio");
  updateVadUI("IDLE", 0);
  console.log("[AudioCapture] Captura detenida y recursos liberados.");
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

// Handle VAD mode checkbox change
if (vadModeToggle) {
  vadModeToggle.addEventListener("change", () => {
    if (!isLocalRecording && recordText) {
      recordText.textContent = vadModeToggle.checked ? "Iniciar Grabación (Manos Libres)" : "Grabar Turno (Manual)";
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

