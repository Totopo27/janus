# Janus (Interview Edition): Plan de Arquitectura y Flujo de Trabajo

> **Sistema Local-First para Entrevistas Multi-Hablante con Ingesta Móvil, Diarización Robusta y Minutas Adaptativas.**
> *Hardware: RTX 3060 12GB + 64GB RAM + Cliente Web Móvil (WSS/PWA)*

---

## 1. Visión General y Arquitectura del Sistema

```text
[MÓVIL (Entrevistador)]
   │
   ├── (Micrófono Web Audio API: 16kHz PCM / Opus)
   ▼
[CAPA DE INGESTA (FastAPI / WSS / TLS Local)]
   │
   ├── (Jitter Buffer desacoplado + Conversión Float32)
   ▼
[PIPELINE DE PROCESAMIENTO (RTX 3060 / CPU)]
   │
   ├── 1. Silero VAD (Filtro de silencios / actividad)
   ├── 2. Sherpa-ONNX / Whisper (ASR -> texto + timestamps de palabras)
   ├── 3. Nemotron Sortformer / PyAnnote (Diarización -> matriz de hablantes y overlaps)
   └── 4. Fusión Conversacional & Filtro Anti-Phantoms (Asigna texto a Speaker_00, 01, etc.)
   │
   ├── (Streaming WSS en vivo hacia el móvil: subtítulos con orador)
   ▼
[POST-PROCESAMIENTO & LLM ADAPTATIVO]
   │
   ├── Factory Router (Ollama Local / Claude / Gemini / OpenAI)
   ├── Extracción estructurada:
   │    ├── Minuta ejecutiva
   │    ├── Temas clave discutidos
   │    ├── Acuerdos y compromisos por participante
   │    └── Análisis de preguntas y respuestas de la entrevista
   ▼
[BASE DE DATOS & EXPORTACIÓN]
   └── SQLite (FTS5) + Exportación Markdown / PDF / JSON
```

---

## 2. Ecosistema de Skills Modulares del Proyecto

Ubicación: `.opencode/skills/`

| Skill | Ubicación | Rol y Responsabilidad |
| :--- | :--- | :--- |
| **`janus-audio-transport`** | `.opencode/skills/janus-audio-transport/` | Ingesta remota desde el móvil vía WebSockets, manejo de buffers de audio, decodificación PCM y estabilidad de red. |
| **`janus-diarization-core`** | `.opencode/skills/janus-diarization-core/` | Modelos de diarización (Sortformer / Sherpa-ONNX), manejo de solapamientos (overlap speech) y eliminación de hablantes fantasmas. |
| **`janus-llm-engine`** | `.opencode/skills/janus-llm-engine/` | Capa adaptativa multi-proveedor (Ollama local RTX 3060, Claude, Gemini) y prompts estructurados para minutas de entrevistas. |
| **`janus-interview-ui`** | `.opencode/skills/janus-interview-ui/` | Frontend web responsivo para móvil: visualización en vivo de la entrevista, control de grabación, edición colaborativa de transcripción/nombres de oradores y manipulación interactiva de minutas y notas. |

---

## 3. Flujo de Trabajo Secuencial con Compuertas de Validación (Gates)

```text
[FASE 0] Auditoría de Cimientos & Limpieza de Scope       ────► Gate 0: Tests existentes en verde (100% pass) [APROBADO]
   │
   ▼
[FASE 1] Ingesta Remota Móvil (WebSockets + Web Audio)    ────► Gate 1: Stream de audio desde móvil recibido sin pérdida por 2 min [APROBADO]
   │
   ▼
[FASE 2] Motor de Diarización & Fusión Anti-Overlap       ────► Gate 2: Distinción clara de 3-4 interlocutores con solapamiento simulado [APROBADO]
   │
   ▼
[FASE 3] Capa LLM Adaptativa & Generador de Minutas       ────► Gate 3: Minuta estructurada generada con Ollama (local) y API externa [APROBADO]
   │
   ▼
[FASE 4] Integración Móvil End-to-End & Teleprompter Live ────► Gate 4: Entrevista completa en vivo grabada desde móvil con minuta final [APROBADO]
```

---

### Detalle de Fases y Compuertas de Salida (Gates)

#### FASE 0: Auditoría de Cimientos & Limpieza de Scope
* **Objetivo:** Asegurar que el core actual de Janus no sufra regresiones y preparar la arquitectura hexagonal para el caso de uso de entrevistas.
* **Skill responsable:** `project-architect`
* **Gate 0 (Medible):** Suite de pruebas de pytest ejecutándose al 100% sin advertencias críticas y archivo de configuración preparado para los nuevos perfiles de diarización.

#### FASE 1: Ingesta Remota Móvil (Audio Streaming)
* **Objetivo:** Permitir que cualquier teléfono móvil en la misma red local (o vía túnel seguro) transmita el audio del micrófono en tiempo real al servidor en tu PC.
* **Skill responsable:** `janus-audio-transport`
* **Gate 1 (Medible):** Enlace WebSocket funcional que mantenga una transmisión continua de audio de al menos 2 minutos desde el navegador del móvil a 16kHz PCM, almacenando el stream sin desbordamiento de búfer.

#### FASE 2: Motor de Diarización & Fusión de Conversación
* **Objetivo:** Procesar el audio recibido identificando con precisión quién habla (2, 3 o 4 participantes), resolviendo interrupciones y filtrando falsos hablantes.
* **Skill responsable:** `janus-diarization-core`
* **Gate 2 (Medible):** Ejecución de una prueba automatizada con archivo de audio multi-hablante que compruebe una precisión de separación de oradores con < 15% de error de asignación y cero "hablantes fantasmas" de duración < 2 segundos.

#### FASE 3: Capa LLM Adaptativa & Generación de Minutas
* **Objetivo:** Conectar el transcript etiquetado con el motor de IA seleccionado (Ollama usando tu RTX 3060 con Qwen 2.5 14B, o Gemini/Claude en la nube).
* **Skill responsable:** `janus-llm-engine`
* **Gate 3 (Medible):** Generación exitosa de un JSON y Markdown estructurado con: Resumen, Preguntas & Respuestas por participante, Acuerdos y Próximos Pasos, evaluado tanto en local (Ollama) como con proveedor externo.

#### FASE 4: Experiencia Móvil Completa, Teleprompter & Editor Interactivo
* **Objetivo:** Unificar la interfaz del móvil para que el entrevistador vea en directo la transcripción con oradores, pueda renombrar etiquetas (ej. cambiar `Speaker_01` por "Juan Pérez"), corregir texto sobre la marcha, y disparar/editar la minuta interactiva con regeneración adaptativa por LLM.
* **Skill responsable:** `janus-interview-ui`
* **Gate 4 (Medible):** Simulación completa de una entrevista en vivo de 3 minutos ejecutada desde el móvil que produzca la minuta final editable, con persistencia inmediata de cambios en SQLite y exportación en < 30 segundos tras presionar "Finalizar".
