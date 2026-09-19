import logging
from typing import List, Optional
from janus.domain.models import ChatMessage, Meeting
from janus.ports.storage_port import IMeetingRepository
from janus.ports.llm_port import ILLMProvider

logger = logging.getLogger(__name__)
MAX_MEETING_CONTEXT_CHARS = 100_000
MAX_QUESTION_CHARS = 4_000
MAX_HISTORY_MESSAGES = 20


class MeetingChatService:
    """
    Service that facilitates intelligent conversational Q&A over meeting transcripts,
    executive summaries, and action items using BYOM (Ollama or Gemini).
    """

    def __init__(self, repository: IMeetingRepository, llm_provider: ILLMProvider) -> None:
        self.repository = repository
        self.llm_provider = llm_provider

    def build_meeting_context(self, meeting: Meeting) -> str:
        lines = []
        lines.append(f"Título: {meeting.title}")
        lines.append(f"Estado: {meeting.status}")
        if meeting.topic_key:
            lines.append(f"Tema / Topic: {meeting.topic_key}")
        lines.append(f"Participante A: {meeting.speaker_a.name} ({meeting.speaker_a.native_language})")
        lines.append(f"Participante B: {meeting.speaker_b.name} ({meeting.speaker_b.native_language})")

        if meeting.summary:
            lines.append("\n[RESUMEN EJECUTIVO]")
            lines.append(meeting.summary.executive_summary)
            if meeting.summary.key_points:
                lines.append("\nPuntos Clave:")
                for pt in meeting.summary.key_points:
                    lines.append(f"- {pt}")
            if meeting.summary.action_items:
                lines.append("\nCompromisos / Tareas:")
                for itm in meeting.summary.action_items:
                    status = "Hecho" if itm.completed else "Pendiente"
                    hint = f" (Plazo: {itm.due_hint})" if itm.due_hint else ""
                    lines.append(f"- [{status}] {itm.assignee}: {itm.task}{hint}")

        lines.append("\n[TRANSCRIPCIÓN COMPLETA]")
        if not meeting.turns:
            lines.append("(No hay turnos registrados en esta reunión)")
        else:
            for turn in meeting.turns:
                spk = meeting.get_speaker(turn.speaker_id)
                name = spk.name if spk else turn.speaker_id
                lines.append(
                    f"{name} ({turn.original_transcription.language}): "
                    f"{turn.original_transcription.text} "
                    f"| Traducción ({turn.translation.target_lang}): {turn.translation.translated_text}"
                )

        context = "\n".join(lines)
        if len(context) > MAX_MEETING_CONTEXT_CHARS:
            context = (
                context[:20_000]
                + "\n\n[TRANSCRIPT TRUNCATED FOR SAFETY]\n\n"
                + context[-80_000:]
            )
        return context

    def ask(
        self,
        meeting_id: str,
        question: str,
        history: Optional[List[ChatMessage]] = None,
    ) -> str:
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise ValueError("Question must contain between 1 and 4000 characters")
        if history and len(history) > MAX_HISTORY_MESSAGES:
            raise ValueError("Chat history cannot contain more than 20 messages")

        meeting = self.repository.get_meeting(meeting_id)
        if not meeting:
            raise ValueError(f"Reunión '{meeting_id}' no encontrada en el repositorio.")

        context = self.build_meeting_context(meeting)
        return self.llm_provider.chat_with_meeting(
            meeting_context=context,
            history=history or [],
            question=question,
        )
