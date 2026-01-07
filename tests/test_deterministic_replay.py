import json
from pathlib import Path
import pytest
from agenwatch._kernel.agent import Agent

@pytest.mark.asyncio
async def test_replay_has_no_side_effects(tmp_path):
    # Create fake replay log
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    replay_log = logs_dir / "exec_001.json"

    replay_log.write_text(json.dumps({
    "execution_id": "test-exec-001",
    "started_at": "2025-01-01T00:00:00",
    "ended_at": None,
    "events": [],
    "agent_config": {},
    "success": True,
    "error_message": None
}))


    agent = Agent(
        user_id="test-user",
        execution_mode="replay",
        replay_log_path=replay_log,
    )

    # Should not raise, should not execute anything
    await agent.run("hello")


