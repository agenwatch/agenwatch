"""Pytest configuration for async tests."""
import pytest


# Configure pytest-asyncio mode
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()



