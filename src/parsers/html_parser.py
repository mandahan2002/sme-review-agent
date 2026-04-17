from __future__ import annotations

from typing import List

from bs4 import BeautifulSoup, Tag

from .base_parser import BaseParser, ParsedContent


class HTMLParser(BaseParser):
    def parse(self, content: str) -> ParsedContent:
        soup = BeautifulSoup(content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        title_elem = soup.find(["h1", "title"])
        title = title_elem.get_text(strip=True) if title_elem else "Untitled"

        full_text = soup.get_text(separator=" ", strip=True)
        sections = self._get_sections(soup)

        return ParsedContent(
            title=title,
            full_text=full_text,
            sections=sections,
            format="html",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _get_sections(soup: BeautifulSoup) -> List[dict]:
        sections: List[dict] = []
        for heading in soup.find_all(["h1", "h2", "h3"]):
            heading_text = heading.get_text(strip=True)
            parts: List[str] = []
            for sibling in heading.find_next_siblings():
                if isinstance(sibling, Tag) and sibling.name in {"h1", "h2", "h3"}:
                    break
                parts.append(sibling.get_text(strip=True))
            body = " ".join(parts).strip()
            if body:
                sections.append({"title": heading_text, "content": body})
        if not sections:
            sections = [{"title": "Main Content", "content": soup.get_text(strip=True)}]
        return sections
