"""Playwright Chromium page reader.

Only navigates and reads: no clicks, logins or form submissions. One browser is started per
run and reused for every page (and later for PDF rendering).
"""

import io
import logging
import re
import time
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

from app.config import Settings
from app.schemas import PageContent

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200
TRANSIENT_NET_ERRORS = (
    "ERR_CONNECTION_RESET", "ERR_CONNECTION_CLOSED", "ERR_TIMED_OUT",
    "ERR_NETWORK_CHANGED", "ERR_EMPTY_RESPONSE", "ERR_HTTP2_PROTOCOL_ERROR",
)

# Reads metadata and main text in the page. Removing boilerplate from the live DOM keeps
# innerText's line breaks, which a detached clone would lose.
EXTRACT_JS = r"""
() => {
  const meta = (names) => {
    for (const n of names) {
      const m = document.querySelector(`meta[property="${n}"], meta[name="${n}"], meta[itemprop="${n}"]`);
      if (m && m.content && m.content.trim()) return m.content.trim();
    }
    return null;
  };
  const fromJsonLd = (key) => {
    for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const data = JSON.parse(s.textContent);
        const items = Array.isArray(data) ? data : (data['@graph'] || [data]);
        for (const it of items) if (it && typeof it[key] === 'string') return it[key];
      } catch (e) {}
    }
    return null;
  };
  const bodyText = document.body ? document.body.innerText : '';
  const visible = (label) => {
    const m = bodyText.match(new RegExp('(?:' + label + ')\\s*:?\\s*(\\d{1,2} [A-Z][a-z]+ \\d{4})'));
    return m ? m[1] : null;
  };
  // Labelled definition lists, e.g. Ofgem's <dt>Publication date:</dt><dd><time datetime="...">
  const labelled = (pattern) => {
    for (const dt of document.querySelectorAll('dt')) {
      if (!pattern.test(dt.textContent)) continue;
      const dd = dt.nextElementSibling;
      if (!dd || dd.tagName !== 'DD') continue;
      const time = dd.querySelector('time[datetime]');
      const value = time ? time.getAttribute('datetime') : dd.textContent.trim();
      if (value) return value;
    }
    return null;
  };
  const published =
    meta(['article:published_time', 'datePublished', 'dcterms.issued', 'DC.date.issued', 'govuk:first-published-at', 'dcterms.date', 'DC.date'])
    || fromJsonLd('datePublished') || labelled(/publication date|published/i) || visible('Published|Publication date');
  const updated =
    meta(['article:modified_time', 'dateModified', 'dcterms.modified', 'govuk:public-updated-at'])
    || fromJsonLd('dateModified') || labelled(/last updated|updated|modified/i) || visible('Last updated');

  const root = document.querySelector('main, article, [role="main"]') || document.body;
  let text = '';
  if (root) {
    root.querySelectorAll('script, style, noscript, nav, footer, header, aside, form, iframe, svg, [aria-hidden="true"]')
      .forEach((el) => el.remove());
    text = root.innerText || '';
  }
  return {
    title: document.title || '',
    site: meta(['og:site_name']),
    published, updated, text,
  };
}
"""


def clean_text(text: str) -> str:
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def describe_dates(published: str | None, updated: str | None) -> str | None:
    parts = []
    if published:
        parts.append(f"published {published}")
    if updated and updated != published:
        parts.append(f"last updated {updated}")
    return "; ".join(parts) or None


class Browser:
    def __init__(self, settings: Settings, sleep=time.sleep):
        self.timeout_ms = settings.page_timeout_ms
        self.max_attempts = settings.browser_max_retries
        # The browser keeps up to the raw limit; the researcher selects what the model sees.
        self.char_limit = settings.page_raw_char_limit
        self._sleep = sleep
        self._pw = self._browser = self._context = None

    def __enter__(self) -> "Browser":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(accept_downloads=False)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for closer in (self._context, self._browser):
            if closer is not None:
                try:
                    closer.close()
                except PlaywrightError:
                    pass
        if self._pw is not None:
            self._pw.stop()
        self._pw = self._browser = self._context = None

    @property
    def context(self):
        if self._context is None:
            raise RuntimeError("Browser is not started; use it as a context manager")
        return self._context

    def open_page(self, url: str) -> PageContent:
        for attempt in range(1, self.max_attempts + 1):
            page, retryable = self._attempt(url)
            if page.ok or not retryable or attempt == self.max_attempts:
                return page
            log.warning("browser: transient failure for %s (%s), retrying", url, page.error)
            self._sleep(2 ** (attempt - 1))
        raise AssertionError("unreachable")

    # ---- internals ----

    def _attempt(self, url: str) -> tuple[PageContent, bool]:
        """Returns the page result and whether a failure is worth retrying."""
        page = self.context.new_page()
        try:
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            except PlaywrightTimeoutError:
                return self._failed(url, f"navigation timed out after {self.timeout_ms // 1000}s"), True
            except PlaywrightError as exc:
                message = str(exc).splitlines()[0]
                if "Download is starting" in message:  # Chromium downloads PDFs instead of rendering them
                    return self._fetch_document(url), False
                return self._failed(url, message), any(code in message for code in TRANSIENT_NET_ERRORS)

            if response is None:
                return self._failed(url, "no response received"), True
            status = response.status
            if status >= 400:
                return self._failed(url, f"HTTP {status}", status), status >= 500

            content_type = (response.headers.get("content-type") or "").lower()
            if "pdf" in content_type:
                return self._pdf_page(url, response.body(), status, page.url), False
            if content_type and not any(t in content_type for t in ("html", "text/plain", "xml")):
                return self._failed(url, f"unsupported content type: {content_type.split(';')[0]}", status), False

            try:
                page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightTimeoutError:
                pass  # some pages never go idle; the DOM is already loaded
            info = page.evaluate(EXTRACT_JS)
            text = clean_text(info["text"])
            if len(text) < MIN_TEXT_CHARS:
                return self._failed(url, "page has too little readable text", status), False
            return PageContent(
                url=url,
                final_url=page.url,
                ok=True,
                title=info["title"].strip() or url,
                text=text[: self.char_limit],
                http_status=status,
                published_date=describe_dates(info["published"], info["updated"]),
                publisher=info["site"] or urlparse(page.url).netloc,
            ), False
        except PlaywrightError as exc:
            return self._failed(url, f"browser error: {str(exc).splitlines()[0]}"), False
        finally:
            page.close()

    def _fetch_document(self, url: str) -> PageContent:
        try:
            response = self.context.request.get(url, timeout=self.timeout_ms)
        except PlaywrightError as exc:
            return self._failed(url, f"download failed: {str(exc).splitlines()[0]}")
        if not response.ok:
            return self._failed(url, f"HTTP {response.status}", response.status)
        if "pdf" not in response.headers.get("content-type", "").lower():
            return self._failed(url, "unsupported download (not a PDF)", response.status)
        return self._pdf_page(url, response.body(), response.status, response.url)

    def _pdf_page(self, url: str, data: bytes, status: int, final_url: str) -> PageContent:
        try:
            reader = PdfReader(io.BytesIO(data))
            chunks, total = [], 0
            for pdf_page in reader.pages:
                chunk = pdf_page.extract_text() or ""
                chunks.append(chunk)
                total += len(chunk)
                if total >= self.char_limit:
                    break
            title = (reader.metadata.title if reader.metadata else None) or ""
        except Exception as exc:  # pypdf raises many types on malformed files
            return self._failed(url, f"could not read PDF: {exc}", status)
        text = clean_text("\n".join(chunks))
        if len(text) < MIN_TEXT_CHARS:
            return self._failed(url, "PDF has too little extractable text", status)
        return PageContent(
            url=url,
            final_url=final_url,
            ok=True,
            content_type="pdf",
            title=title.strip() or urlparse(final_url).path.rsplit("/", 1)[-1] or url,
            text=text[: self.char_limit],
            http_status=status,
            publisher=urlparse(final_url).netloc,
        )

    @staticmethod
    def _failed(url: str, error: str, status: int | None = None) -> PageContent:
        return PageContent(url=url, ok=False, http_status=status, error=error)
