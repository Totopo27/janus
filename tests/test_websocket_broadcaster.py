import asyncio
import json
from unittest.mock import AsyncMock
from janus.adapters.transport.websocket_broadcaster import WebSocketBroadcaster
from janus.domain.events import TranscriptionCompletedEvent


def test_websocket_broadcaster_lifecycle():
    async def run_test():
        broadcaster = WebSocketBroadcaster()
        session_id = "sess_test_1"

        ws_mock = AsyncMock()
        ws_mock.send_text = AsyncMock()

        # 1. Connect
        await broadcaster.connect(session_id, ws_mock)
        assert session_id in broadcaster._connections
        assert ws_mock in broadcaster._connections[session_id]

        # 2. Broadcast raw
        test_payload = {"msg": "hello"}
        await broadcaster.broadcast_raw(session_id, test_payload)
        ws_mock.send_text.assert_called_once_with(json.dumps(test_payload))

        # 3. Broadcast domain event
        event = TranscriptionCompletedEvent(
            session_id=session_id,
            speaker_id="spk_a",
            text="Buenos días",
            language="es",
        )
        await broadcaster.broadcast_event(session_id, event)
        assert ws_mock.send_text.call_count == 2
        last_call_arg = ws_mock.send_text.call_args[0][0]
        parsed = json.loads(last_call_arg)
        assert parsed["event_name"] == "TranscriptionCompleted"
        assert parsed["data"]["text"] == "Buenos días"

        # 4. Disconnect
        await broadcaster.disconnect(session_id, ws_mock)
        assert session_id not in broadcaster._connections

    asyncio.run(run_test())


def test_websocket_broadcaster_handles_dead_socket():
    async def run_test():
        broadcaster = WebSocketBroadcaster()
        session_id = "sess_test_dead"

        failing_ws = AsyncMock()
        failing_ws.send_text.side_effect = ConnectionResetError("Client disconnected")

        await broadcaster.connect(session_id, failing_ws)
        assert len(broadcaster._connections[session_id]) == 1

        # Should not raise exception, but clean up dead socket
        await broadcaster.broadcast_raw(session_id, {"ping": "pong"})
        assert len(broadcaster._connections.get(session_id, set())) == 0

    asyncio.run(run_test())
