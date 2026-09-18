from datetime import datetime, timezone
import logging
import re
from typing import List, Optional
from janus.domain.models import Meeting, MeetingSummary, ActionItem
from janus.ports.storage_port import IMeetingRepository

logger = logging.getLogger(__name__)


class MeetingNotesService:
    """
    Generates Zoom-style structured meeting notes, executive summaries,
    and action items from recorded bilingual conversations.
    """

    def __init__(self, repository: IMeetingRepository) -> None:
        self.repo = repository

    def generate_notes_and_finalize(self, meeting_id: str) -> Optional[MeetingSummary]:
        """
        Analyzes meeting dialogue turns, builds structured notes & action items,
        and saves the summary to the database marking the meeting as completed.
        """
        meeting = self.repo.get_meeting(meeting_id)
        if not meeting:
            logger.warning(f"Meeting '{meeting_id}' not found.")
            return None

        turns = meeting.turns
        if not turns:
            summary = MeetingSummary(
                executive_summary="La reunión no registró intervenciones de audio.",
                key_points=[],
                action_items=[],
            )
            self.repo.save_summary(meeting_id, summary)
            return summary

        # 1. Executive Summary Synthesis
        spk_a_name = meeting.speaker_a.name
        spk_b_name = meeting.speaker_b.name
        num_turns = len(turns)

        executive_summary = (
            f"Conversación bilingüe entre {spk_a_name} ({meeting.speaker_a.native_language.upper()}) "
            f"y {spk_b_name} ({meeting.speaker_b.native_language.upper()}), compuesta por {num_turns} "
            f"intervenciones traducidas simultáneamente con Janus. Los participantes coordinaron puntos técnicos "
            f"y establecieron compromisos de trabajo."
        )

        # 2. Key Discussion Points Extraction
        key_points: List[str] = []
        for t in turns:
            text = t.original_transcription.text.strip()
            speaker = meeting.get_speaker(t.speaker_id)
            name = speaker.name if speaker else t.speaker_id
            if len(text) > 10 and not any(p.endswith(text) for p in key_points):
                key_points.append(f"{name}: {text}")
            if len(key_points) >= 5:
                break

        # 3. Action Items Extraction (Commitment Detection)
        action_items: List[ActionItem] = []
        commitment_patterns_es = [
            r"(?:yo me comprometo a|me comprometo a|voy a|yo voy a|me encargo de|yo haré|revisaré|prepararé)\s+(.+)",
        ]
        commitment_patterns_en = [
            r"(?:i will|i'll|i commit to|i can take care of|i'm going to|let me handle)\s+(.+)",
        ]

        for t in turns:
            text_es = t.original_transcription.text if t.original_transcription.language == "es" else t.translation.translated_text
            text_en = t.original_transcription.text if t.original_transcription.language == "en" else t.translation.translated_text
            speaker = meeting.get_speaker(t.speaker_id)
            assignee = speaker.name if speaker else t.speaker_id

            found_task = None
            for pat in commitment_patterns_es:
                m = re.search(pat, text_es, re.IGNORECASE)
                if m:
                    found_task = m.group(1).strip().capitalize()
                    break

            if not found_task:
                for pat in commitment_patterns_en:
                    m = re.search(pat, text_en, re.IGNORECASE)
                    if m:
                        found_task = m.group(1).strip().capitalize()
                        break

            if found_task:
                # Deduce due date hint if present
                due_hint = None
                if any(w in found_task.lower() for w in ["mañana", "tomorrow"]):
                    due_hint = "Mañana"
                elif any(w in found_task.lower() for w in ["viernes", "friday"]):
                    due_hint = "Viernes"
                elif any(w in found_task.lower() for w in ["lunes", "monday"]):
                    due_hint = "Lunes"

                action_items.append(
                    ActionItem(
                        assignee=assignee,
                        task=found_task,
                        due_hint=due_hint,
                    )
                )

        # Fallback action item if none detected
        if not action_items and turns:
            action_items.append(
                ActionItem(
                    assignee=spk_a_name,
                    task="Revisar acuerdos y seguimiento de la sesión.",
                )
            )

        summary = MeetingSummary(
            executive_summary=executive_summary,
            key_points=key_points,
            action_items=action_items,
        )

        self.repo.save_summary(meeting_id, summary)
        logger.info(f"Meeting notes and summary saved for meeting '{meeting_id}'.")
        return summary

    def export_as_markdown(self, meeting_id: str) -> str:
        """
        Exports the meeting notes, action items, and bilingual transcript
        into a clean, publication-ready GitHub Flavored Markdown document.
        """
        meeting = self.repo.get_meeting(meeting_id)
        if not meeting:
            return "# Reunión no encontrada"

        created_dt = datetime.fromtimestamp(meeting.created_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_badge = "🟢 Completada" if meeting.status == "completed" else "🟡 En Progreso"

        lines = [
            f"# 📋 Minuta de Reunión: {meeting.title}",
            "",
            f"> **Fecha y Hora**: {created_dt}  ",
            f"> **Estado**: {status_badge}  ",
            f"> **Participantes**: {meeting.speaker_a.name} ({meeting.speaker_a.native_language}) ↔️ {meeting.speaker_b.name} ({meeting.speaker_b.native_language})  ",
            "",
            "---",
            "",
            "## 📌 Resumen Ejecutivo",
            "",
            meeting.summary.executive_summary if meeting.summary else "Pendiente de finalización.",
            "",
            "## 🔑 Puntos Clave",
            "",
        ]

        if meeting.summary and meeting.summary.key_points:
            for pt in meeting.summary.key_points:
                lines.append(f"- {pt}")
        else:
            lines.append("- No se registraron puntos clave.")

        lines.extend([
            "",
            "## ✅ Compromisos y Tareas (Action Items)",
            "",
        ])

        if meeting.summary and meeting.summary.action_items:
            for item in meeting.summary.action_items:
                check = "[x]" if item.completed else "[ ]"
                due = f" *(Fecha límite: {item.due_hint})*" if item.due_hint else ""
                lines.append(f"- {check} **{item.assignee}**: {item.task}{due}")
        else:
            lines.append("- No hay tareas pendientes registradas.")

        lines.extend([
            "",
            "---",
            "",
            "## 📜 Transcripción Bilingüe Cronológica",
            "",
            "| Tiempo | Locutor | Idioma Original | Traducción Simultánea |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for t in meeting.turns:
            speaker = meeting.get_speaker(t.speaker_id)
            speaker_name = speaker.name if speaker else t.speaker_id
            time_str = datetime.fromtimestamp(t.created_at, tz=timezone.utc).strftime("%H:%M:%S")
            orig = t.original_transcription.text.replace("|", "\\|")
            trans = t.translation.translated_text.replace("|", "\\|")
            lines.append(f"| {time_str} | **{speaker_name}** | {orig} | {trans} |")

        lines.append("")
        return "\n".join(lines)
