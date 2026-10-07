"""IEEE Xplore link detection, metadata access, and Discord rendering."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import discord

from bot.modules.arxiv import ABSTRACT_LIMIT, generate_tldr
from bot.utils import DeletableView, RetryView, render_math


IEEE_URL_RE = re.compile(
    r"https?://(?:www\.)?ieeexplore\.ieee\.org/(?:abstract/)?document/(\d+)",
    re.IGNORECASE,
)
IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
IEEE_API_KEY = os.environ.get("IEEE_API_KEY", "")
PAPER_CACHE_TTL = timedelta(days=14)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IEEEArticle:
    article_number: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    publication_date: str
    doi: str
    url: str


def _cache_directory() -> Path:
    return Path(os.environ.get("PIPXIV_CACHE_DIR", "data/cache")) / "ieee"


def _clear_expired_cache(directory: Path) -> None:
    cutoff = datetime.now(timezone.utc) - PAPER_CACHE_TTL
    for metadata_path in directory.glob("*.json"):
        modified = datetime.fromtimestamp(metadata_path.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            metadata_path.unlink(missing_ok=True)
            metadata_path.with_suffix(".pdf").unlink(missing_ok=True)


def _load_cached_paper(article_number: str) -> tuple[IEEEArticle, bytes | None] | None:
    directory = _cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    _clear_expired_cache(directory)
    metadata_path = directory / f"{article_number}.json"
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        article = IEEEArticle(
            article_number=str(data["article_number"]),
            title=str(data["title"]),
            abstract=str(data["abstract"]),
            authors=tuple(data["authors"]),
            publication_date=str(data["publication_date"]),
            doi=str(data["doi"]),
            url=str(data["url"]),
        )
        pdf_path = metadata_path.with_suffix(".pdf")
        pdf = pdf_path.read_bytes() if pdf_path.exists() else None
        if pdf is not None and not pdf.startswith(b"%PDF"):
            pdf_path.unlink(missing_ok=True)
            pdf = None
        return article, pdf
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        metadata_path.unlink(missing_ok=True)
        metadata_path.with_suffix(".pdf").unlink(missing_ok=True)
        return None


def _save_cached_paper(article: IEEEArticle, pdf: bytes | None) -> None:
    directory = _cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{article.article_number}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "article_number": article.article_number,
                "title": article.title,
                "abstract": article.abstract,
                "authors": article.authors,
                "publication_date": article.publication_date,
                "doi": article.doi,
                "url": article.url,
            }
        ),
        encoding="utf-8",
    )
    if pdf is not None:
        metadata_path.with_suffix(".pdf").write_bytes(pdf)


def extract_ieee_article_number(content: str) -> str | None:
    """Return the first IEEE Xplore article number in a message."""
    match = IEEE_URL_RE.search(content)
    return match.group(1) if match else None


def _authors(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names = []
    for author in value:
        if isinstance(author, dict):
            name = author.get("full_name") or author.get("name")
        else:
            name = author
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return tuple(names)


def parse_ieee_response(payload: bytes, article_number: str) -> IEEEArticle:
    """Parse one IEEE Metadata API response."""
    result = json.loads(payload)
    articles = result.get("articles", [])
    if not articles:
        raise ValueError("IEEE returned no paper for that article number")

    article = articles[0]
    resolved_number = str(article.get("article_number") or article_number)
    return IEEEArticle(
        article_number=resolved_number,
        title=str(article.get("title") or resolved_number),
        abstract=str(article.get("abstract") or ""),
        authors=_authors(article.get("authors")),
        publication_date=str(article.get("publication_date") or article.get("publication_year") or ""),
        doi=str(article.get("doi") or ""),
        url=f"https://ieeexplore.ieee.org/document/{resolved_number}",
    )


def fetch_article(article_number: str) -> IEEEArticle:
    if not IEEE_API_KEY:
        raise RuntimeError("IEEE_API_KEY is required")
    query = urlencode({"apikey": IEEE_API_KEY, "article_number": article_number, "format": "json"})
    request = Request(f"{IEEE_API_URL}?{query}", headers={"User-Agent": "Pipxiv/1.0"})
    with urlopen(request, timeout=15) as response:
        return parse_ieee_response(response.read(), article_number)


def article_message_content(content: str, article: IEEEArticle) -> str:
    """Replace the submitted IEEE URL while preserving the user's context."""
    match = IEEE_URL_RE.search(content)
    article_link = discord.utils.escape_markdown(article.title)
    if not match:
        return article_link
    return content[: match.start()] + article_link + content[match.end() :]


def article_embed(article: IEEEArticle, author: discord.abc.User, tldr: str | None = None) -> discord.Embed:
    embed = discord.Embed(
        title=article.title[:256],
        url=article.url,
        description=render_math(
            article.abstract[: ABSTRACT_LIMIT - 3] + ("..." if len(article.abstract) > ABSTRACT_LIMIT else "")
        ),
        color=discord.Color.blue(),
    )
    embed.set_author(name=f"IEEE:{article.article_number}")
    if article.authors:
        embed.add_field(name="Authors", value=", ".join(article.authors)[:1024], inline=False)
    if article.publication_date:
        embed.add_field(name="Published", value=article.publication_date)
    if tldr:
        embed.add_field(name="TLDR", value=render_math(tldr), inline=False)
    embed.set_footer(text=f"Posted by {author.display_name}")
    return embed


class IEEEModule:
    """Handle IEEE Xplore document links."""

    def matches(self, content: str) -> bool:
        return extract_ieee_article_number(content) is not None

    async def handle(self, message: discord.Message) -> None:
        article_number = extract_ieee_article_number(message.content)
        if not article_number:
            return

        try:
            cached = await asyncio.to_thread(_load_cached_paper, article_number)
            if cached:
                article, _ = cached
            else:
                article = await asyncio.to_thread(fetch_article, article_number)
                await asyncio.to_thread(_save_cached_paper, article, None)
            try:
                tldr = await asyncio.to_thread(generate_tldr, article.abstract)
            except Exception:
                LOGGER.exception("Failed to generate TLDR for IEEE article %s", article_number)
                tldr = None
            await message.channel.send(
                content=article_message_content(message.content, article),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await message.channel.send(
                embed=article_embed(article, message.author, tldr),
                view=DeletableView(article.url, message.author.id),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await message.delete()
        except (json.JSONDecodeError, ValueError):
            await message.reply(
                "I couldn't find that IEEE paper.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
        except Exception:
            LOGGER.exception("Failed to process IEEE article %s", article_number)
            await message.reply(
                "I couldn't retrieve that IEEE paper right now.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
