import pytest
from helpers import make_settings

from app.tools.browser import Browser


@pytest.fixture(scope="session")
def chromium():
    """One real headless Chromium for PDF tests (no network needed)."""
    try:
        browser = Browser(make_settings()).__enter__()
    except Exception as exc:  # Chromium not installed
        pytest.skip(f"Playwright Chromium unavailable: {exc}")
    yield browser
    browser.close()
