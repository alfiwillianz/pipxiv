import asyncio
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from bot.modules.arxiv import LLM_TIMEOUT, extract_arxiv_id, generate_tldr, paper_message_content, parse_api_response
from bot.modules.crossref import extract_doi, parse_crossref_response, work_message_content
from bot.modules.elsevier import extract_elsevier_doi, extract_elsevier_pii, parse_elsevier_response
from bot.modules.ieee import (
    _load_cached_paper,
    _save_cached_paper,
    article_message_content,
    extract_ieee_article_number,
    parse_ieee_response,
)
from bot.client import WebLinkBot
from bot.utils import DISCORD_MAX_UPLOAD_BYTES, guard_discord_upload, render_math


ATOM_RESPONSE = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>  A useful paper  </title>
    <summary> An abstract. </summary>
    <published>2024-01-20T12:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <category term="cs.LG" />
    <category term="cs.CL" />
  </entry>
</feed>'''

IEEE_RESPONSE = b'''{"articles":[{"article_number":"1234567","title":"An IEEE Paper",
"abstract":"A useful abstract.","authors":[{"full_name":"Ada Lovelace"}],
"publication_date":"2026-01-01","doi":"10.1109/TEST.1234567"}]}'''

CROSSREF_RESPONSE = b'''{"message":{"DOI":"10.1038/nphys1170","title":["Measured measurement"],
"abstract":"<jats:p>A useful <jats:b>abstract</jats:b>.</jats:p>",
"author":[{"given":"Markus","family":"Aspelmeyer"}],
"published":{"date-parts":[[2009,1]]}}}'''

ELSEVIER_RESPONSE = b'''<abstracts-retrieval-response xmlns:dc="http://purl.org/dc/elements/1.1/">
<coredata><dc:identifier>doi:10.1016/j.test.2026.1</dc:identifier>
<dc:title>Elsevier Paper</dc:title><prism:coverDate xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/">2026-01-01</prism:coverDate>
<dc:description>Useful abstract.</dc:description></coredata>
<authors><author><ce:given-name xmlns:ce="http://www.elsevier.com/xml/common/dtd">Ada</ce:given-name>
<ce:surname xmlns:ce="http://www.elsevier.com/xml/common/dtd">Lovelace</ce:surname></author></authors>
</abstracts-retrieval-response>'''


class ArxivTests(unittest.TestCase):
    def test_discord_upload_guard_allows_files_at_or_below_limit(self):
        payload = b"x" * DISCORD_MAX_UPLOAD_BYTES
        self.assertIs(guard_discord_upload(payload, module="test", item="small"), payload)
        self.assertIsNone(guard_discord_upload(None, module="test", item="missing"))

    def test_discord_upload_guard_omits_oversized_files(self):
        with self.assertLogs("bot.utils", level="WARNING") as logs:
            result = guard_discord_upload(
                b"x" * (DISCORD_MAX_UPLOAD_BYTES + 1), module="test", item="large"
            )
        self.assertIsNone(result)
        self.assertIn("above Discord's upload limit", logs.output[0])

    def test_extracts_modern_and_pdf_links(self):
        self.assertEqual(extract_arxiv_id("see https://arxiv.org/abs/2401.12345v2"), "2401.12345v2")
        self.assertEqual(extract_arxiv_id("https://arxiv.org/pdf/2401.12345.pdf"), "2401.12345")

    def test_rejects_non_arxiv_urls(self):
        self.assertIsNone(extract_arxiv_id("https://example.com/abs/2401.12345"))

    def test_parses_atom_metadata(self):
        paper = parse_api_response(ATOM_RESPONSE, "2401.12345")
        self.assertEqual(paper.title, "A useful paper")
        self.assertEqual(paper.authors, ("Ada Lovelace",))
        self.assertEqual(paper.categories, ("cs.LG", "cs.CL"))
        self.assertEqual(paper.abs_url, "https://arxiv.org/abs/2401.12345v2")

    @patch("bot.modules.arxiv.urlopen")
    def test_generates_tldr_from_router_response(self, mock_urlopen):
        response = unittest.mock.Mock()
        response.__enter__ = lambda value: response
        response.__exit__ = unittest.mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"A concise summary."}}]}'
        mock_urlopen.return_value = response

        self.assertEqual(generate_tldr("The abstract describes a method."), "A concise summary.")
        payload = mock_urlopen.call_args.args[0].data
        self.assertIn(b'"temperature": 0.1', payload)
        self.assertIn(b'"stream": false', payload)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], LLM_TIMEOUT)

    def test_skips_tldr_for_empty_abstract(self):
        self.assertIsNone(generate_tldr(""))

    def test_renders_inline_latex_as_unicode_without_rewriting_prose(self):
        rendered = render_math(r"The bound is $k \geq \log^2 n$.")
        self.assertEqual(rendered, "The bound is 𝑘≥log²𝑛.")

    @patch.dict(
        "bot.utils.os.environ",
        {"LLM_BASE_URL": "http://llm.test/v1/chat/completions", "LLM_MODEL": "router/fallback"},
        clear=False,
    )
    @patch("bot.utils.urlopen")
    def test_uses_env_llm_for_unsupported_math(self, mock_urlopen):
        response = unittest.mock.Mock()
        response.__enter__ = lambda value: response
        response.__exit__ = unittest.mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"Converted x"}}]}'
        mock_urlopen.return_value = response

        self.assertEqual(render_math(r"The result is $\\custom{x}$."), "Converted x")
        self.assertEqual(mock_urlopen.call_args.args[0].full_url, "http://llm.test/v1/chat/completions")

    def test_replaces_url_and_preserves_context(self):
        paper = parse_api_response(ATOM_RESPONSE, "2401.12345")
        content = "https://arxiv.org/abs/2401.12345v2 lorem ipsum"
        self.assertEqual(
            paper_message_content(content, paper),
            "A useful paper lorem ipsum",
        )

    def test_parses_ieee_metadata_and_preserves_context(self):
        self.assertEqual(
            extract_ieee_article_number("read https://ieeexplore.ieee.org/document/1234567"),
            "1234567",
        )
        article = parse_ieee_response(IEEE_RESPONSE, "1234567")
        self.assertEqual(article.authors, ("Ada Lovelace",))
        self.assertEqual(
            article_message_content("https://ieeexplore.ieee.org/document/1234567 lorem ipsum", article),
            "An IEEE Paper lorem ipsum",
        )

    def test_expires_cached_papers_after_two_weeks(self):
        article = parse_ieee_response(IEEE_RESPONSE, "1234567")
        with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
            "bot.modules.ieee.os.environ", {"PIPXIV_CACHE_DIR": cache_dir}, clear=False
        ):
            _save_cached_paper(article, None)
            self.assertIsNotNone(_load_cached_paper("1234567"))
            metadata_path = os.path.join(cache_dir, "ieee", "1234567.json")
            old = time.time() - (15 * 24 * 60 * 60)
            os.utime(metadata_path, (old, old))
            self.assertIsNone(_load_cached_paper("1234567"))
            self.assertFalse(os.path.exists(metadata_path))

    def test_parses_crossref_metadata_and_preserves_context(self):
        doi = "10.1038/nphys1170"
        self.assertEqual(extract_doi("read https://doi.org/" + doi), doi)
        work = parse_crossref_response(CROSSREF_RESPONSE, doi)
        self.assertEqual(work.abstract, "A useful abstract .")
        self.assertEqual(work.authors, ("Markus Aspelmeyer",))
        self.assertEqual(
            work_message_content("https://doi.org/10.1038/nphys1170 lorem ipsum", work),
            "Measured measurement lorem ipsum",
        )

    def test_parses_elsevier_abstract_metadata(self):
        doi = "10.1016/j.test.2026.1"
        self.assertEqual(extract_elsevier_doi("https://doi.org/" + doi), doi)
        self.assertEqual(
            extract_elsevier_pii(
                "https://www.sciencedirect.com/science/article/abs/pii/S1568494626015255"
            ),
            "S1568494626015255",
        )
        self.assertIsNone(extract_elsevier_doi("https://doi.org/10.1038/nphys1170"))
        work = parse_elsevier_response(ELSEVIER_RESPONSE, doi)
        self.assertEqual(work.title, "Elsevier Paper")
        self.assertEqual(work.abstract, "Useful abstract.")
        self.assertEqual(work.authors, ("Ada Lovelace",))


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_link_is_rejected_while_processing(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def handle(_message):
            started.set()
            await release.wait()

        module = Mock()
        module.matches.return_value = True
        module.handle = handle
        bot = WebLinkBot([module])
        bot.process_commands = AsyncMock()

        status = SimpleNamespace(delete=AsyncMock())
        first = SimpleNamespace(
            id=1,
            content="https://example.com/paper",
            author=SimpleNamespace(bot=False),
            mentions=[],
            channel=SimpleNamespace(id=10),
            reply=AsyncMock(return_value=status),
        )
        duplicate_reply = AsyncMock()
        duplicate = SimpleNamespace(**{**vars(first), "reply": duplicate_reply, "id": 2})
        bot._delete_later = AsyncMock()

        task = asyncio.create_task(bot.on_message(first))
        await started.wait()
        await bot.on_message(duplicate)
        duplicate_reply.assert_awaited_once()
        self.assertIn("already processing", duplicate_reply.await_args.args[0])
        await asyncio.sleep(0)
        bot._delete_later.assert_awaited_once_with(duplicate_reply.return_value, 10)

        release.set()
        await task
        status.delete.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
