# Recursos y Referencias Técnicas: Janus (Interview Edition)

> Documento base de especificaciones oficiales, repositorios de referencia, utilidades y modelos para el sistema de entrevistas en vivo con diarización y minutas.

---

### 1. Runtimes de Inferencia de Voz y Diarización On-Device

* **Sherpa-ONNX (Next-gen Kaldi / k2-fsa)** ([GitHub](https://github.com/k2-fsa/sherpa-onnx) | [Diarization Docs](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html))
  * **Descripción:** Runtime ligero C++ y Python sobre ONNX Runtime. Soporta VAD (Silero), ASR (Whisper, Zipformer) y Diarización Offline/Streaming (PyAnnote Segmentation + 3D-Speaker/CAM++ embeddings con clustering rápido).
  * **Uso principal:** Motor de transcripción y segmentación de audio en CPU/GPU con bajo consumo de VRAM y cero dependencia de compilaciones complejas de PyTorch.

* **NVIDIA NeMo / Sortformer (Nemotron-3 Diarization)** ([NeMo Diarization Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/ASR_with_SpeakerDiarization.ipynb))
  * **Descripción:** Arquitectura basada en transformers para diarización multi-hablante que detecta y resuelve *overlap speech* (habla simultánea) directamente a nivel de frames para hasta 4-8 hablantes.
  * **Uso principal:** Resolver entrevistas caóticas donde 2 o más personas se interrumpen al mismo tiempo.

* **Silero VAD (ONNX)** ([GitHub](https://snakers4.github.io/silero-vad/))
  * **Descripción:** Detector de actividad de voz de grado industrial (<2MB de peso, <1ms de latencia en CPU).
  * **Uso principal:** Puerta de descarte de silencios antes de enviar fragmentos pesados a los encoders ASR y Diarización.

---

### 2. Ecosistema de Referencia en Asistentes de Reuniones Open-Source

* **TranscrIA** ([GitHub](https://github.com/Martossien/transcria))
  * **Descripción:** Portal de transcripción y minutas en servidor GPU/CPU local.
  * **Uso principal:** Referencia para arquitecturas de colas de procesamiento, alineación de timestamps SRT con etiquetas de hablantes y pipeline de limpieza de transcripciones antes de pasar al LLM.

* **Meetily (Meeting Minutes)** ([GitHub](https://github.com/Zackriya-Solutions/meetily))
  * **Descripción:** Asistente de reuniones local de escritorio (Rust + Web).
  * **Uso principal:** Patrones de captura y flujo de minutas con Ollama y modelos locales.

* **ab-transcript & hark** ([ab-transcript](https://github.com/effixis/ab-transcript) | [hark](https://github.com/fossabot/hark))
  * **Descripción:** Proyectos de captura dual y asignación de hablantes basada en solapamiento de intervalos temporales (*timestamp interval overlap*).
  * **Uso principal:** Lógica matemática de fusión para asociar cada bloque de texto reconocido al hablante correcto.

---

### 3. Ingesta de Audio Remoto y Web PWA / WebSocket

* **Web Audio API & MediaRecorder API (W3C)** ([MDN Web Audio](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API))
  * **Descripción:** APIs nativas de navegadores móviles para capturar audio de micrófono en PCM de 16kHz o fragmentos comprimidos WebM/Opus.
  * **Uso principal:** El cliente móvil no necesita instalar apps nativas complejas; se conecta por HTTPS/WSS a la IP de la máquina anfitriona y transmite el audio del micrófono de la entrevista en tiempo real.

* **FastAPI WebSockets & Audio Stream Buffers** ([FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/))
  * **Descripción:** Transporte asíncrono para streaming bidireccional: recepción de chunks binarios de audio y emisión en tiempo real de transcripción + orador identificado hacia la pantalla del móvil.

---

### 4. Capa Adaptativa de Modelos de Lenguaje (LLM Engines)

* **Ollama Local API** ([Ollama Docs](https://github.com/ollama/ollama/blob/main/docs/api.md))
  * **Descripción:** Servidor local OpenAI-compatible para correr modelos en la RTX 3060 12GB (ej. Qwen 2.5 14B, Llama 3.1 8B, MiniCPM).
  * **Uso principal:** Procesamiento 100% privado y sin costo de generación de minutas, compromisos y análisis de la entrevista.

* **Anthropic Claude API & Google Gemini API**
  * **Descripción:** Motores en la nube para entrevistas complejas o largas donde se requiera ventana de contexto de cientos de miles de tokens o máxima fidelidad analítica.
  * **Uso principal:** Adaptadores opcionales configurables vía variables de entorno / API key según la necesidad del usuario.
