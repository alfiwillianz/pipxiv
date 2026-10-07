"""Discord helpers shared by web-link modules."""

from __future__ import annotations

import re
import json
import logging
import os
import unicodedata
from collections.abc import Awaitable, Callable
from urllib.request import Request, urlopen

import discord
from unicodeitplus import replace as replace_unicode_math


MATH_SPAN_RE = re.compile(r"\$(?!\$)(.+?)(?<!\\)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
LOGGER = logging.getLogger(__name__)

# Keep uploads below Discord's unboosted 10 MiB limit to leave room for
# multipart request overhead and avoid rejecting the entire message with 413.
DISCORD_MAX_UPLOAD_BYTES = 9 * 1024 * 1024


def guard_discord_upload(data: bytes | None, *, module: str, item: str) -> bytes | None:
    """Return an upload if it fits Discord's limit, otherwise log and omit it."""
    if data is not None and len(data) > DISCORD_MAX_UPLOAD_BYTES:
        LOGGER.warning(
            "%s %s attachment is %.1f MB, above Discord's upload limit; posting without it",
            module,
            item,
            len(data) / (1024 * 1024),
        )
        return None
    return data


def render_math(text: str) -> str:
    """Render delimited LaTeX as Unicode without changing surrounding prose."""
    def convert(match: re.Match[str]) -> str:
        source = next(group for group in match.groups() if group is not None)
        source = re.sub(r"\\widetilde\{([^{}]+)\}", r"\\tilde \1", source)
        source = re.sub(r"\\text\{([^{}]+)\}", r"\1", source)
        try:
            converted = replace_unicode_math(source)
            return re.sub(r"\\(log|ln|exp|max|min)", r"\1", converted)
        except Exception:
            return source

    rendered = MATH_SPAN_RE.sub(convert, text)
    needs_fallback = any(char in rendered for char in ("\\", "{", "}", "^")) or any(
        unicodedata.combining(char) for char in rendered
    )
    if rendered == text or not needs_fallback:
        return rendered

    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not base_url or not model:
        return rendered
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/chat/completions"):
        base_url += "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Convert only LaTeX math to readable Unicode. Preserve all prose and mathematical "
                    "meaning exactly. Return only the converted text."
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request = Request(base_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read())
        fallback = str(result["choices"][0]["message"]["content"]).strip()
        return fallback or rendered
    except Exception:
        return rendered


class DeletableView(discord.ui.View):
    """View with a link button and a permission-aware delete button."""

    def __init__(self, url: str, source_author_id: int) -> None:
        super().__init__(timeout=None)
        self.source_author_id = source_author_id
        self.add_item(discord.ui.Button(label="Open", style=discord.ButtonStyle.link, url=url))

    def can_delete(self, user: discord.abc.User) -> bool:
        if user.id == self.source_author_id:
            return True
        return isinstance(user, discord.Member) and user.guild_permissions.manage_messages

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.danger, custom_id="web-link:delete")
    async def delete_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.can_delete(interaction.user):
            await interaction.response.send_message(
                "Only the original poster or a member with Manage Messages can delete this.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()
        if interaction.message:
            await interaction.message.delete()


class RetryView(discord.ui.View):
    """Offer to rerun a failed web-link request."""

    def __init__(self, retry: Callable[[], Awaitable[None]]) -> None:
        super().__init__(timeout=300)
        self._retry = retry
        self.retry_button = discord.ui.Button(label="Try Again", style=discord.ButtonStyle.primary)
        self.retry_button.callback = self._retry_callback
        self.add_item(self.retry_button)

    async def _retry_callback(self, interaction: discord.Interaction) -> None:
        self.retry_button.disabled = True
        await interaction.response.edit_message(view=self)
        try:
            await self._retry()
        finally:
            if interaction.message:
                await interaction.message.delete()
