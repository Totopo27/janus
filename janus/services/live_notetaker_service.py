import json
import logging
import re
import time
from typing import Dict, List, Optional
from janus.domain.models import LiveMeetingNotes, ActionItem, ConversationTurn
from janus.domain.events import LiveNotesUpdatedEvent
from janus.ports.storage_port import IMeetingRepository
from janus.ports.llm_port import ILLMProvider
from janus.ports.broadcaster_port import IEventBroadcaster

logger = logging.getLogger(__name__)


class LiveNotetakerService:
    """
    Continuous real-time meeting notetaker modeled after Zoom AI Companion.
    Listens incrementally to dialogue turns, updates current topics,
    extracts key takeaways, tracks action items, and generates live catch-ups.
    """

    def __init__(
        self,
        repository: IMeetingRepository,
        llm_provider: ILLMProvider,
        broadcaster: Optional[IEventBroadcaster] = None,
        batch_size: int = 3,
    ) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.broadcaster = broadcaster
        self.batch_size = batch_size
        self._cache: Dict[str, LiveMeetingNotes] = {}
        self._pending_turns_count: Dict[str, int] = {}

    def get_live_notes(self, meeting_id: str) -> LiveMeetingNotes:
        """Returns the current live notes for a meeting, creating a default state if empty."""
        if meeting_id not in self._cache:
            self._cache[meeting_id] = LiveMeetingNotes(meeting_id=meeting_id)
        return self._cache[meeting_id]

    def process_turn(
        self,
        meeting_id: str,
        turn: ConversationTurn,
    ) -> Optional[LiveMeetingNotes]:
        """
        Receives a newly completed conversation turn. When batch_size turns have
        accumulated, triggers incremental AI reasoning and emits a live notes update.
        """
        count = self._pending_turns_count.get(meeting_id, 0) + 1
        self._pending_turns_count[meeting_id] = count

        if count >= self.batch_size:
            return self.update_notes(meeting_id)
        return None

    def update_notes(self, meeting_id: str) -> LiveMeetingNotes:
        """Forces an immediate synthesis update of live notes using all conversation turns so far."""
        meeting = self.repository.get_meeting(meeting_id)
        if not meeting or not meeting.turns:
            return self.get_live_notes(meeting_id)

        current_notes = self.get_live_notes(meeting_id)
        recent_turns = meeting.turns[-10:]  # Context of up to the last 10 turns

        dialogue_text = "\n".join([
            f"- {meeting.get_speaker(t.speaker_id).name if meeting.get_speaker(t.speaker_id) else t.speaker_id}: {t.original_transcription.text} (Traducción: {t.translation.translated_text})"
            for t in recent_turns
        ])

        system_prompt = (
            "Eres el asistente inteligente Zoom AI Companion para esta reunión. "
            "Tu trabajo es escuchar activamente la conversación y redactar notas ejecutivas en vivo. "
            "Debes responder ÚNICAMENTE con un JSON válido que tenga la siguiente estructura exacta:\n"
            "{\n"
            '  "current_topic": "Título breve del tema que se está discutiendo ahora",\n'
            '  "key_takeaways": ["Punto clave 1", "Punto clave 2"],\n'
            '  "action_items": [{"assignee": "Nombre", "task": "Tarea específica", "due_hint": "Plazo o null"}]\n'
            "}"
        )

        user_prompt = (
            f"Notas previas:\n"
            f"- Tema actual: {current_notes.current_topic}\n"
            f"- Puntos acumulados: {len(current_notes.key_takeaways)}\n\n"
            f"Últimos turnos de la reunión:\n{dialogue_text}\n\n"
            "Actualiza las notas ejecutivas, el tema en curso y los nuevos compromisos detectados en formato JSON:"
        )

        try:
            raw_response = self.llm_provider.generate(prompt=user_prompt, system_prompt=system_prompt)
            parsed_data = self._parse_json_response(raw_response)

            if parsed_data:
                if parsed_data.get("current_topic"):
                    current_notes.current_topic = parsed_data["current_topic"]

                # Merge new takeaways without exact duplicate strings
                for pt in parsed_data.get("key_takeaways", []):
                    if pt and pt not in current_notes.key_takeaways:
                        current_notes.key_takeaways.append(pt)

                # Merge action items
                for itm in parsed_data.get("action_items", []):
                    if isinstance(itm, dict) and itm.get("task"):
                        action_item = ActionItem(
                            assignee=itm.get("assignee", "Participante"),
                            task=itm.get("task", ""),
                            due_hint=itm.get("due_hint"),
                        )
                        # Check duplicate
                        if not any(a.task.lower() == action_item.task.lower() for a in current_notes.action_items):
                            current_notes.action_items.append(action_item)

                current_notes.updated_at = time.time()
                current_notes.last_processed_turn_index = len(meeting.turns)

                # Broadcast live event to all connected WebSockets
                if self.broadcaster:
                    event = LiveNotesUpdatedEvent(
                        meeting_id=meeting_id,
                        current_topic=current_notes.current_topic,
                        key_takeaways=current_notes.key_takeaways,
                        action_items=[
                            {
                                "assignee": a.assignee,
                                "task": a.task,
                                "due_hint": a.due_hint,
                                "completed": a.completed,
                            }
                            for a in current_notes.action_items
                        ],
                    )
                    try:
                        import asyncio
                        coro = self.broadcaster.broadcast_event(meeting_id, event)
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(coro)
                        except RuntimeError:
                            # No running loop — broadcast_event was already called (sync path)
                            pass
                    except Exception as broadcast_err:
                        logger.warning("Live notes broadcast failed: %s", broadcast_err)

        except Exception as e:
            logger.error("Failed to update live notes with LLM: %s", e)

        self._pending_turns_count[meeting_id] = 0
        return current_notes

    def catch_up(self, meeting_id: str, last_n_turns: int = 6) -> str:
        """
        Zoom AI Companion 'Catch Me Up' feature.
        Generates an instant 2-3 sentence executive recap of what happened in the recent turns.
        """
        meeting = self.repository.get_meeting(meeting_id)
        if not meeting or not meeting.turns:
            return "Aún no se ha registrado suficiente conversación para ponerse al día."

        recent_turns = meeting.turns[-last_n_turns:]
        dialogue = "\n".join([
            f"{meeting.get_speaker(t.speaker_id).name if meeting.get_speaker(t.speaker_id) else t.speaker_id}: {t.original_transcription.text}"
            for t in recent_turns
        ])

        system_prompt = (
            "Eres Zoom AI Companion. Un participante pide ponerse al día (Catch Me Up). "
            "Responde en 2 o 3 oraciones concisas, directas y enérgicas resumiendo qué se discutió, "
            "qué decisiones se tomaron y si se asignó alguna tarea."
        )
        user_prompt = f"Diálogo reciente de la reunión:\n{dialogue}\n\n¿Qué me perdí?"

        try:
            return self.llm_provider.generate(prompt=user_prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.error("Error generating catch up summary: %s", e)
            return f"No se pudo generar el resumen en este momento ({e})."

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Robustly extracts JSON from raw LLM responses (stripping markdown code fences)."""
        if not text:
            return None
        clean = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)

        # Attempt direct JSON parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Search for first { ... } block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from LLM response: %s", text[:200])
        return None
