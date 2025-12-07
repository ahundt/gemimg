"""Pytest configuration and fixtures for gemimg tests."""

import pytest


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "test-api-key-12345"
