"""Elsevier DOI link detection and abstract retrieval."""

from __future__ import annotations

import asyncio
import logging
import re
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen
import urllib.error

import discord

from bot.modules.arxiv import ABSTRACT_LIMIT, generate_tldr
from bot.modules.crossref import DOI_URL_RE, extract_doi
from bot.utils import DeletableView, RetryView, render_math


ELSEVIER_API_URL = "https://api.elsevier.com/content/abstract/doi/{}"
ELSEVIER_PII_API_URL = "https://api.elsevier.com/content/abstract/pii/{}"
ELSEVIER_API_KEY = os.environ.get("ELSEVIER_API_KEY", "")
ELSEVIER_DOI_PREFIXES = ("10.1016/", "10.1006/")
ELSEVIER_PII_URL_RE = re.compile(
    r"https?://(?:www\.)?sciencedirect\.com/science/article/(?:abs/)?pii/([A-Z0-9]+)",
    re.IGNORECASE,
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElsevierWork:
    doi: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    published: str
    url: str


def extract_elsevier_doi(content: str) -> str | None:
    doi = extract_doi(content)
    return doi if doi and doi.lower().startswith(ELSEVIER_DOI_PREFIXES) else None


def extract_elsevier_pii(content: str) -> str | None:
    match = ELSEVIER_PII_URL_RE.search(content)
    return match.group(1) if match else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _first(root: ET.Element, names: set[str]) -> str:
    for element in root.iter():
        if _local_name(element.tag) in names:
            value = _text(element)
            if value:
                return value
    return ""


def parse_elsevier_response(payload: bytes, requested_doi: str) -> ElsevierWork:
    """Parse Elsevier Abstract Retrieval XML."""
    root = ET.fromstring(payload)
    doi = _first(root, {"doi"}) or requested_doi
    title = _first(root, {"title"}) or doi
    abstract = _first(root, {"description", "abstracttext"})
    authors = []
    for author in root.iter():
        if _local_name(author.tag) != "author":
            continue
        name = " ".join(
            value
            for child in author.iter()
            if _local_name(child.tag) in {"given-name", "surname", "indexed-name"}
            if (value := _text(child))
        )
        if name and name not in authors:
            authors.append(name)
    published = _first(root, {"coverDate", "publicationdate", "publication-year"})
    return ElsevierWork(
        doi=doi,
        title=title,
        abstract=abstract,
        authors=tuple(authors[:20]),
        published=published,
        url=f"https://doi.org/{doi}",
    )


def fetch_work(identifier: str, identifier_type: str = "doi") -> ElsevierWork:
    if not ELSEVIER_API_KEY:
        raise RuntimeError("ELSEVIER_API_KEY is required")
    api_url = ELSEVIER_API_URL if identifier_type == "doi" else ELSEVIER_PII_API_URL
    url = f"{api_url.format(quote(identifier, safe=''))}?view=META_ABS"
    request = Request(
        url,
        headers={
            "Accept": "application/xml",
            "X-ELS-APIKey": ELSEVIER_API_KEY,
            "User-Agent": "Pipxiv/1.0",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return parse_elsevier_response(response.read(), identifier)
    except urllib.error.HTTPError as exc:
        # Many keys are entitled only to basic metadata, not abstracts.
        if exc.code not in (401, 403):
            raise
        fallback = Request(
            f"{api_url.format(quote(identifier, safe=''))}?view=META",
            headers={
                "Accept": "application/xml",
                "X-ELS-APIKey": ELSEVIER_API_KEY,
                "User-Agent": "Pipxiv/1.0",
            },
        )
        with urlopen(fallback, timeout=20) as response:
            return parse_elsevier_response(response.read(), identifier)


def work_message_content(content: str, work: ElsevierWork) -> str:
    match = DOI_URL_RE.search(content) or ELSEVIER_PII_URL_RE.search(content)
    link = discord.utils.escape_markdown(work.title)
    if not match:
        return link
    return content[: match.start()] + link + content[match.end() :]


def work_embed(work: ElsevierWork, author: discord.abc.User, tldr: str | None = None) -> discord.Embed:
    embed = discord.Embed(
        title=work.title[:256],
        url=work.url,
        description=render_math(work.abstract[: ABSTRACT_LIMIT - 3] + ("..." if len(work.abstract) > ABSTRACT_LIMIT else ""))
        if work.abstract
        else "Abstract is not available for this identifier with the configured Elsevier API key.",
        color=discord.Color.orange(),
    )
    embed.set_author(name=f"Elsevier DOI:{work.doi}")
    if work.authors:
        embed.add_field(name="Authors", value=", ".join(work.authors)[:1024], inline=False)
    if work.published:
        embed.add_field(name="Published", value=work.published)
    if tldr:
        embed.add_field(name="TLDR", value=render_math(tldr), inline=False)
    embed.set_footer(text=f"Posted by {author.display_name}")
    return embed


class ElsevierModule:
    """Handle Elsevier DOI links through the Abstract Retrieval API."""

    def matches(self, content: str) -> bool:
        return extract_elsevier_doi(content) is not None or extract_elsevier_pii(content) is not None

    async def handle(self, message: discord.Message) -> None:
        doi = extract_elsevier_doi(message.content)
        pii = extract_elsevier_pii(message.content)
        if not doi and not pii:
            return

        try:
            work = await asyncio.to_thread(fetch_work, doi or pii, "doi" if doi else "pii")
            try:
                tldr = await asyncio.to_thread(generate_tldr, work.abstract)
            except Exception:
                LOGGER.exception("Failed to generate TLDR for Elsevier reference %s", doi or pii)
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
        except (ET.ParseError, ValueError):
            await message.reply(
                "I couldn't find that Elsevier paper.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
        except Exception:
            LOGGER.exception("Failed to process Elsevier reference %s", doi or pii)
            await message.reply(
                "I couldn't retrieve that Elsevier paper right now.",
                mention_author=False,
                view=RetryView(lambda: self.handle(message)),
            )
