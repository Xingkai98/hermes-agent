"""Regression: the heartbeat poller must *start* a turn when idle, not queue it.

Issue #85119: a Telegram session heartbeat (``/heartbeat every 1m <prompt>``)
documented that missed ticks coalesce — "one heartbeat turn, not a backlog" —
but the poller injected due prompts via ``_enqueue_fifo``, which only writes to
the adapter's pending-message FIFO. An idle session has no consumer for that
FIFO, so every due tick piled up and the next inbound message replayed the
whole backlog as separate turns.

The fix routes the idle fire through ``deliver_wake`` (→ ``adapter.handle_message``
with ``internal=True``), so an idle session actually runs the turn and the
``due_prompt`` gate governs the next fire.  This test pins that contract: a due
heartbeat for an idle session must reach ``handle_message``, never just sit in a
queue.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def _build_runner(monkeypatch, tmp_path) -> GatewayRunner:
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    return GatewayRunner(GatewayConfig())


@pytest.mark.asyncio
async def test_heartbeat_poller_delivers_via_handle_message(monkeypatch, tmp_path):
    """A due heartbeat for an idle session is delivered as a real turn."""
    runner = _build_runner(monkeypatch, tmp_path)

    handle_message = AsyncMock()
    adapter = SimpleNamespace(handle_message=handle_message)
    source = SimpleNamespace(platform="telegram", chat_id="chat-1")
    quick_key = "agent:main:telegram:dm:chat-1"
    session_id = "sess-1"

    runner._heartbeat_watch = {quick_key: (source, session_id)}
    runner._running_agents = {}
    runner._adapter_for_source = lambda src: adapter

    class _DueHeartbeat:
        def has_heartbeat(self) -> bool:
            return True

        def due_prompt(self):
            return "PROMPT"

    import hermes_cli.heartbeat as hbmod

    monkeypatch.setattr(hbmod, "HeartbeatManager", lambda session_id=None: _DueHeartbeat())
    # Drive the poll loop at full speed instead of the real 5s cadence.
    monkeypatch.setattr(hbmod, "POLL_SECONDS", 0)

    runner._start_heartbeat_poller()
    # Let the poll task run a few iterations.
    for _ in range(5):
        await asyncio.sleep(0)

    # Clean up the forever-loop before asserting.
    poll_task = runner._heartbeat_poll_task
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    assert handle_message.await_count >= 1
    delivered = handle_message.await_args.args[0]
    assert delivered.text == "PROMPT"
    assert delivered.internal is True
