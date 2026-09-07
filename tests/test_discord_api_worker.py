import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from discord_api_worker import DiscordAPIWorker


def api_state(**overrides):
    defaults = {
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.8,
        "presence_penalty": 0.5,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_blocking_inference_does_not_block_discord_event_loop():
    started = threading.Event()
    inference_thread_ids = []

    async def blocking_call(**kwargs):
        inference_thread_ids.append(threading.get_ident())
        started.set()
        time.sleep(0.15)
        return kwargs["user_content"]

    worker = DiscordAPIWorker(blocking_call, api_state(), name="heartbeat-test")

    async def scenario():
        discord_thread_id = threading.get_ident()
        request = asyncio.create_task(worker.call_api(user_content="finished"))
        while not started.is_set():
            await asyncio.sleep(0)

        ticks = 0
        deadline = asyncio.get_running_loop().time() + 0.05
        while asyncio.get_running_loop().time() < deadline:
            ticks += 1
            await asyncio.sleep(0.005)

        assert not request.done()
        assert ticks >= 3
        assert await request == "finished"
        return discord_thread_id

    try:
        discord_thread_id = asyncio.run(scenario())
    finally:
        worker.shutdown()

    assert inference_thread_ids == [worker.thread_id]
    assert inference_thread_ids[0] != discord_thread_id


def test_requests_are_serialized_and_defaults_are_snapshotted_at_submission():
    state = api_state(temperature=0.2)
    calls = []
    active = 0
    max_active = 0

    async def tracked_call(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        calls.append((kwargs["user_content"], kwargs["temperature"]))
        await asyncio.sleep(0.02)
        active -= 1
        return kwargs["user_content"]

    worker = DiscordAPIWorker(tracked_call, state, name="serialization-test")
    first = worker.submit(user_content="first")
    state.temperature = 0.4
    second = worker.submit(user_content="second")
    state.temperature = 1.0

    async def collect():
        return await asyncio.gather(
            asyncio.wrap_future(first),
            asyncio.wrap_future(second),
        )

    try:
        assert asyncio.run(collect()) == ["first", "second"]
    finally:
        worker.shutdown()

    assert calls == [("first", 0.2), ("second", 0.4)]
    assert max_active == 1


def test_explicit_sampling_settings_override_worker_snapshot():
    seen = {}

    async def capture_call(**kwargs):
        seen.update(kwargs)
        return "ok"

    worker = DiscordAPIWorker(capture_call, api_state(temperature=0.2), name="override-test")
    try:
        result = asyncio.run(
            worker.call_api(user_content="hello", temperature=1.3, top_p=0.6)
        )
    finally:
        worker.shutdown()

    assert result == "ok"
    assert seen["temperature"] == 1.3
    assert seen["top_p"] == 0.6


def test_worker_exception_propagates_to_awaiting_caller():
    async def failed_call(**kwargs):
        raise ValueError("inference failed")

    worker = DiscordAPIWorker(failed_call, api_state(), name="failure-test")
    try:
        with pytest.raises(ValueError, match="inference failed"):
            asyncio.run(worker.call_api(user_content="hello"))
    finally:
        worker.shutdown()


def test_worker_rejects_requests_after_shutdown():
    async def unused_call(**kwargs):
        return "unused"

    worker = DiscordAPIWorker(unused_call, api_state(), name="shutdown-test")
    worker.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        worker.submit(user_content="too late")
