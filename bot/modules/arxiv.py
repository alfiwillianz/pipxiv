"""arXiv link detection, API access, and Discord rendering."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import html
import json
import logging
import os
import re
import threading
import time
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen

import discord

from bot.utils import (
    DISCORD_MAX_UPLOAD_BYTES,
    DeletableView,
    RetryView,
    render_math,
)


ARXIV_API_URL = "https://export.arxiv.org/api/query?id_list={}"
ARXIV_URL_RE = re.compile(
    r"https?://(?:export\.)?(?:www\.)?arxiv\.org/(?:abs|pdf)/([^\s?#]+)",
    re.IGNORECASE,
)
ARXIV_ID_RE = re.compile(
    r"(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:20128/v1").rstrip("/")
LLM_URL = (
    LLM_BASE_URL
    if LLM_BASE_URL.endswith("/chat/completions")
    else LLM_BASE_URL + "/chat/completions"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "cx/gpt-6-luna")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-7a1855903ad8e97c-krzh20-f0e15aa5")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))
LLM_TEMPERATURE = 0.1
# Some endpoints sit behind Cloudflare, which 403s the default urllib UA.
LLM_USER_AGENT = os.environ.get(
    "LLM_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
)
ABSTRACT_LIMIT = 3500
PAPER_CACHE_TTL = timedelta(days=14)
ATOM = "{http://www.w3.org/2005/Atom}"
LOGGER = logging.getLogger(__name__)

# Discord hard limits. Messages over 2000 chars fail with 413 (error 40005);
# embeds must total <=6000 chars with each field <=1024 and description <=4096.
DISCORD_CONTENT_LIMIT = 2000
DISCORD_EMBED_TOTAL_LIMIT = 6000
DISCORD_EMBED_DESCRIPTION_LIMIT = 4096
DISCORD_EMBED_FIELD_LIMIT = 1024
# arXiv asks for no more than one request every 3 seconds per IP and will
# 429 (and stay blocked for a couple of minutes) if links are handled
# concurrently without this. fetch_paper run in worker threads
# via asyncio.to_thread, so this needs to be thread-safe, not asyncio-safe.
_ARXIV_MIN_INTERVAL = 3.0
_arxiv_throttle_lock = threading.Lock()
_arxiv_next_allowed_time = 0.0


def _throttle_arxiv_request() -> None:
    """Block the calling thread until it's safe to hit arxiv.org again."""
    global _arxiv_next_allowed_time
    with _arxiv_throttle_lock:
        now = time.monotonic()
        wait = _arxiv_next_allowed_time - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _arxiv_next_allowed_time = now + _ARXIV_MIN_INTERVAL


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    summary: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published: datetime | None
    abs_url: str
    pdf_url: str


def _cache_directory() -> Path:
    return Path(os.environ.get("PIPBOT_CACHE_DIR", "data/cache")) / "arxiv"


def _cache_key(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", arxiv_id)


def _clear_expired_cache(directory: Path) -> None:
    cutoff = datetime.now(timezone.utc) - PAPER_CACHE_TTL
    for metadata_path in directory.glob("*.json"):
        if datetime.fromtimestamp(metadata_path.stat().st_mtime, timezone.utc) < cutoff:
            metadata_path.unlink(missing_ok=True)
            metadata_path.with_suffix(".pdf").unlink(missing_ok=True)


def _load_cached_paper(arxiv_id: str) -> tuple[Paper, bytes | None] | None:
    directory = _cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    _clear_expired_cache(directory)
    metadata_path = directory / f"{_cache_key(arxiv_id)}.json"
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        published = data.get("published")
        paper = Paper(
            arxiv_id=str(data["arxiv_id"]),
            title=str(data["title"]),
            summary=str(data["summary"]),
            authors=tuple(data["authors"]),
            categories=tuple(data["categories"]),
            published=datetime.fromisoformat(published) if published else None,
            abs_url=str(data["abs_url"]),
            pdf_url=str(data["pdf_url"]),
        )
        pdf_path = metadata_path.with_suffix(".pdf")
        pdf = None
        if pdf_path.exists():
            # Don't read oversized files into memory; they can't be posted anyway.
            if pdf_path.stat().st_size > DISCORD_MAX_UPLOAD_BYTES:
                pdf_path.unlink(missing_ok=True)
            else:
                candidate = pdf_path.read_bytes()
                if candidate.startswith(b"%PDF"):
                    pdf = candidate
                else:
                    pdf_path.unlink(missing_ok=True)
        return paper, pdf
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        metadata_path.unlink(missing_ok=True)
        metadata_path.with_suffix(".pdf").unlink(missing_ok=True)
        return None


def _save_cached_paper(paper: Paper, pdf: bytes | None) -> None:
    directory = _cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{_cache_key(paper.arxiv_id)}.json"
    metadata_path.write_text(
        json.dumps({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "summary": paper.summary,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published.isoformat() if paper.published else None,
            "abs_url": paper.abs_url,
            "pdf_url": paper.pdf_url,
        }),
        encoding="utf-8",
    )
    if pdf is not None:
        metadata_path.with_suffix(".pdf").write_bytes(pdf)


def extract_arxiv_id(content: str) -> str | None:
    """Return the first supported arXiv identifier in a Discord message."""
    match = ARXIV_URL_RE.search(content)
    if not match:
        return None

    identifier = match.group(1).removesuffix(".pdf").strip("/")
    return identifier if ARXIV_ID_RE.fullmatch(identifier) else None


def _text(element: ET.Element | None) -> str:
    return html.unescape(" ".join((element.text or "").split())) if element is not None else ""


def parse_api_response(payload: bytes, requested_id: str) -> Paper:
    """Parse one arXiv Atom API response."""
    root = ET.fromstring(payload)
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        raise ValueError("arXiv returned no paper for that identifier")

    raw_id = _text(entry.find(f"{ATOM}id"))
    arxiv_id = raw_id.rsplit("/", 1)[-1] or requested_id
    title = _text(entry.find(f"{ATOM}title")) or requested_id
    summary = _text(entry.find(f"{ATOM}summary"))
    authors = tuple(
        _text(author.find(f"{ATOM}name"))
        for author in entry.findall(f"{ATOM}author")
        if _text(author.find(f"{ATOM}name"))
    )
    categories = tuple(
        category.get("term", "")
        for category in entry.findall(f"{ATOM}category")
        if category.get("term")
    )
    published_text = _text(entry.find(f"{ATOM}published"))
    published = datetime.fromisoformat(published_text.replace("Z", "+00:00")) if published_text else None
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        summary=summary,
        authors=authors,
        categories=categories,
        published=published,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def fetch_paper(arxiv_id: str, attempts: int = 3) -> Paper:
    """Fetch paper metadata from the arXiv API, retrying on 429s and timeouts.

    arXiv's export API rate-limits aggressively and occasionally stalls, so a
    single failed attempt shouldn't surface as an error to the user.
    """
    request = Request(ARXIV_API_URL.format(arxiv_id), headers={"User-Agent": "ArxivEmbedBot/1.0"})
    for attempt in range(attempts):
        _throttle_arxiv_request()
        try:
            with urlopen(request, timeout=20) as response:
                return parse_api_response(response.read(), arxiv_id)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            delay = float(exc.headers.get("Retry-After", 0) or 0) or 2 ** (attempt + 2)
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            delay = 2 ** (attempt + 2)
        LOGGER.warning("arXiv fetch for %s failed (attempt %d/%d), retrying in %.0fs", arxiv_id, attempt + 1, attempts, delay)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _looks_like_meta_commentary(text: str) -> bool:
    """True when the model critiqued the input instead of summarizing it.

    Some backends (notably the `auto-fallback` router) ignore the system
    prompt and reply as if the abstract were a user's draft ("This reads like
    a solid abstract...", "A few observations...", "What works well"). Those
    replies are useless as a TLDR, so we detect and discard them.
    """
    lowered = text.lower()
    markers = (
        "this reads like", "this looks like", "this is a strong", "this is a solid",
        "a few observations", "what works well", "here are some", "i can help",
        "if you're writing", "if you are writing", "if you'd like", "would you like",
        "let me know", "in case you're", "when writing", "a couple of", "some suggestions",
        "consider ", "reviewer or reader", "the abstract is",
    )
    if any(marker in lowered for marker in markers):
        return True
    # A pile of headings/bullets masquerading as prose is also a red flag.
    return lowered.count("**") >= 2


def _extractive_tldr(abstract: str, max_words: int = 35) -> str:
    """Deterministic fallback: the abstract's first sentence, trimmed."""
    first_sentence = re.split(r"(?<=[.!?])\s+", " ".join(abstract.split()), maxsplit=1)[0]
    words = first_sentence.split()
    if len(words) > max_words:
        first_sentence = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return first_sentence[:1024]


def _normalize_tldr(text: str, abstract: str = "") -> str | None:
    """Coerce a model response into a single plain-text paragraph.

    The LLM sometimes ignores the one-sentence instruction and returns an
    essay with headings, bullets, or a preamble. Collapse all of that down to
    one paragraph so the embed always shows a clean TLDR. If the model's reply
    is meta-commentary rather than a summary, fall back to the abstract.
    """
    if not text:
        return _extractive_tldr(abstract) if abstract else None
    # Strip any code-fence backticks (handles both block and inline fences).
    cleaned = re.sub(r"`+", "", text)
    # Drop a leading "TLDR"/"Summary" label, with or without a colon.
    cleaned = re.sub(r"(?i)^\s*(tl;?dr|summary|abstract)\b\s*[:\-–]?\s*", "", cleaned.strip(), count=1)
    # Strip markdown headings, bullets, numbering, and emphasis markers.
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", cleaned)
    cleaned = cleaned.replace("**", "")
    # Collapse every run of whitespace (including newlines) into a single space,
    # which also merges headings/bullets into one paragraph.
    cleaned = " ".join(cleaned.split())
    # Cut off trailing meta-commentary the model sometimes appends.
    for marker in (" If you'd like", " If you are writing", " If you're writing",
                   " Let me know", " Would you like", " I can help"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].rstrip()
    if not cleaned or _looks_like_meta_commentary(cleaned):
        return _extractive_tldr(abstract) if abstract else None
    return cleaned[:1024] or None


def generate_tldr(abstract: str) -> str | None:
    """Generate a short, evidence-bound summary from an abstract."""
    if not abstract:
        return None

    payload = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write one-line TLDRs for scientific papers. Reply with exactly one sentence "
                    "of at most 35 words in a single paragraph. Use only facts explicitly stated in the "
                    "abstract; do not infer results, causes, numbers, or implications. Do not include "
                    "headings, bullet points, lists, labels like 'TLDR:', questions, advice, or any "
                    "preamble or follow-up. Output only the sentence, nothing else."
                ),
            },
            {"role": "user", "content": abstract},
        ],
    }
    request = Request(
        LLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": LLM_USER_AGENT,
            **({"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}),
        },
        method="POST",
    )
    with urlopen(request, timeout=LLM_TIMEOUT) as response:
        result = json.loads(response.read())
    content = result["choices"][0]["message"]["content"]
    return _normalize_tldr(content, abstract)


def _truncate(text: str, limit: int) -> str:
    """Clip text to limit chars, appending an ellipsis when shortened."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def paper_message_content(content: str, paper: Paper) -> str:
    """Replace the submitted arXiv URL while preserving the user's context."""
    match = ARXIV_URL_RE.search(content)
    if not match:
        rendered = f"[{discord.utils.escape_markdown(paper.title)}]({paper.abs_url})"
    else:
        paper_name = discord.utils.escape_markdown(paper.title)
        rendered = content[: match.start()] + paper_name + content[match.end() :]
    # The original message can be arbitrarily long; Discord rejects >2000 chars.
    return _truncate(rendered, DISCORD_CONTENT_LIMIT)


def paper_embed(paper: Paper, author: discord.abc.User, tldr: str | None = None) -> discord.Embed:
    embed = discord.Embed(
        title=paper.title[:256],
        url=paper.abs_url,
        description=render_math(
            paper.summary[: ABSTRACT_LIMIT - 3] + ("..." if len(paper.summary) > ABSTRACT_LIMIT else "")
        ),
        color=discord.Color.dark_red(),
    )
    embed.set_author(name=f"arXiv:{paper.arxiv_id}")
    if paper.authors:
        embed.add_field(
            name="Authors",
            value=_truncate(", ".join(paper.authors), DISCORD_EMBED_FIELD_LIMIT),
            inline=False,
        )
    if paper.published:
        embed.add_field(name="Published", value=paper.published.strftime("%Y-%m-%d"))
    if tldr:
        embed.add_field(name="TLDR", value=render_math(tldr), inline=False)
    embed.set_footer(text=f"Posted by {author.display_name}")
    _fit_embed(embed)
    return embed


def _embed_char_count(embed: discord.Embed) -> int:
    total = len(embed.title or "") + len(embed.description or "")
    total += len(embed.footer.text or "") if embed.footer else 0
    total += len(embed.author.name or "") if embed.author else 0
    for field in embed.fields:
        total += len(field.name) + len(field.value)
    return total


def _fit_embed(embed: discord.Embed) -> None:
    """Shrink the embed until it fits Discord's total/description limits."""
    if embed.description and len(embed.description) > DISCORD_EMBED_DESCRIPTION_LIMIT:
        embed.description = _truncate(embed.description, DISCORD_EMBED_DESCRIPTION_LIMIT)
    # Drop the abstract tail (the least essential part) until the total fits.
    while _embed_char_count(embed) > DISCORD_EMBED_TOTAL_LIMIT and embed.description:
        overflow = _embed_char_count(embed) - DISCORD_EMBED_TOTAL_LIMIT
        new_len = max(0, len(embed.description) - overflow - 3)
        if new_len == len(embed.description):
            break
        embed.description = _truncate(embed.description, new_len)


class ArxivModule:
    """Handle arXiv links; copy this module pattern for another website."""

    def matches(self, content: str) -> bool:
        return extract_arxiv_id(content) is not None

    async def handle(self, message: discord.Message) -> None:
        arxiv_id = extract_arxiv_id(message.content)
        if not arxiv_id:
            return

        try:
            cached = await asyncio.to_thread(_load_cached_paper, arxiv_id)
            if cached:
                paper, _ = cached
            else:
                paper = await asyncio.to_thread(fetch_paper, arxiv_id)
                await asyncio.to_thread(_save_cached_paper, paper, None)
            try:
                tldr = await asyncio.to_thread(generate_tldr, paper.summary)
            except Exception:
                LOGGER.exception("Failed to generate TLDR for arXiv link %s", arxiv_id)
                tldr = None
            await message.channel.send(
                content=paper_message_content(message.content, paper),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await message.channel.send(
                embed=paper_embed(paper, message.author, tldr),
                view=DeletableView(paper.abs_url, message.author.id),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await message.delete()
        except (ET.ParseError, ValueError):
            await message.reply(
                "I couldn't find that arXiv paper.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
        except Exception:
            LOGGER.exception("Failed to process arXiv link %s", arxiv_id)
            await message.reply(
                "I couldn't retrieve that arXiv paper right now.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
