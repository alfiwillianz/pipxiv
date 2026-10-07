"""Crossref DOI link detection, metadata access, and Discord rendering."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen

import discord

from bot.modules.arxiv import ABSTRACT_LIMIT, generate_tldr
from bot.utils import DeletableView, RetryView, render_math


DOI_URL_RE = re.compile(
    r"https?://(?:doi\.org|dx\.doi\.org)/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)
CROSSREF_API_URL = "https://api.crossref.org/works/{}"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossrefWork:
    doi: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str
    url: str


def extract_doi(content: str) -> str | None:
    """Return the first DOI from a doi.org link in a message."""
    match = DOI_URL_RE.search(content)
    return match.group(1).rstrip(".,;:)") if match else None


def _clean_abstract(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(text.split())


def _date(message: dict[str, object]) -> str:
    parts = message.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
        return ""
    return "-".join(str(part) for part in parts[0])


def parse_crossref_response(payload: bytes, requested_doi: str) -> CrossrefWork:
    """Parse one Crossref REST API work response."""
    result = json.loads(payload).get("message", {})
    doi = str(result.get("DOI") or requested_doi)
    titles = result.get("title", [])
    title = str(titles[0]) if isinstance(titles, list) and titles else doi
    authors = tuple(
        " ".join(str(author.get(part)) for part in ("given", "family") if author.get(part))
        for author in result.get("author", [])
        if isinstance(author, dict) and (author.get("given") or author.get("family"))
    )
    published = _date(result.get("published", {})) or _date(result.get("issued", {}))
    return CrossrefWork(
        doi=doi,
        title=title,
        abstract=_clean_abstract(result.get("abstract")),
        authors=authors,
        published=published,
        url=f"https://doi.org/{doi}",
    )


def fetch_work(doi: str) -> CrossrefWork:
    request = Request(
        CROSSREF_API_URL.format(quote(doi, safe="")),
        headers={"User-Agent": "Pipxiv/1.0 (mailto:crossref@pipxiv.local)"},
    )
    with urlopen(request, timeout=15) as response:
        return parse_crossref_response(response.read(), doi)


def work_message_content(content: str, work: CrossrefWork) -> str:
    """Replace the submitted DOI URL while preserving the user's context."""
    match = DOI_URL_RE.search(content)
    work_link = discord.utils.escape_markdown(work.title)
    if not match:
        return work_link
    return content[: match.start()] + work_link + content[match.end() :]


def work_embed(work: CrossrefWork, author: discord.abc.User, tldr: str | None = None) -> discord.Embed:
    embed = discord.Embed(
        title=work.title[:256],
        url=work.url,
        description=render_math(
            work.abstract[: ABSTRACT_LIMIT - 3] + ("..." if len(work.abstract) > ABSTRACT_LIMIT else "")
        ),
        color=discord.Color.green(),
    )
    embed.set_author(name=f"DOI:{work.doi}")
    if work.authors:
        embed.add_field(name="Authors", value=", ".join(work.authors)[:1024], inline=False)
    if work.published:
        embed.add_field(name="Published", value=work.published)
    if tldr:
        embed.add_field(name="TLDR", value=render_math(tldr), inline=False)
    embed.set_footer(text=f"Posted by {author.display_name}")
    return embed


class CrossrefModule:
    """Handle DOI links through Crossref metadata."""

    def matches(self, content: str) -> bool:
        return extract_doi(content) is not None

    async def handle(self, message: discord.Message) -> None:
        doi = extract_doi(message.content)
        if not doi:
            return

        try:
            work = await asyncio.to_thread(fetch_work, doi)
            try:
                tldr = await asyncio.to_thread(generate_tldr, work.abstract)
            except Exception:
                LOGGER.exception("Failed to generate TLDR for DOI %s", doi)
                tldr = None
            await message.channel.send(
                content=work_message_content(message.content, work),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await message.channel.send(
                embed=work_embed(work, message.author, tldr),
                view=DeletableView(work.url, message.author.id),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await message.delete()
        except (json.JSONDecodeError, ValueError):
            await message.reply(
                "I couldn't find that DOI paper.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
        except Exception:
            LOGGER.exception("Failed to process DOI %s", doi)
            await message.reply(
                "I couldn't retrieve that DOI paper right now.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
