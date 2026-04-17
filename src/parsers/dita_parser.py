from __future__ import annotations

from typing import List

from lxml import etree

from .base_parser import BaseParser, ParsedContent

_SECTION_TAGS = {
    "section", "concept", "task", "reference",
    "step", "steps", "result", "body", "prereq",
    "context", "postreq", "info", "shortdesc",
}


class DITAParser(BaseParser):
    def parse(self, content: str) -> ParsedContent:
        try:
            root = etree.fromstring(content.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            raise ValueError(f"Invalid XML/DITA: {exc}") from exc

        title = self._get_title(root)
        full_text = " ".join(root.itertext()).strip()
        sections = self._get_sections(root)

        return ParsedContent(
            title=title,
            full_text=full_text,
            sections=sections,
            format="dita",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _local(elem: etree._Element) -> str:
        tag = elem.tag
        return tag.split("}")[-1] if "}" in tag else tag

    def _get_title(self, root: etree._Element) -> str:
        for elem in root.iter():
            if self._local(elem) == "title" and elem.text:
                return elem.text.strip()
        return root.get("id", "Untitled Document")

    def _get_sections(self, root: etree._Element) -> List[dict]:
        sections: List[dict] = []
        for elem in root.iter():
            if self._local(elem) not in _SECTION_TAGS:
                continue
            title = "Content"
            for child in elem:
                if self._local(child) == "title" and child.text:
                    title = child.text.strip()
                    break
            text = " ".join(elem.itertext()).strip()
            if len(text) > 10:
                sections.append({"title": title, "content": text})
        if not sections:
            full = " ".join(root.itertext()).strip()
            sections = [{"title": "Main Content", "content": full}]
        return sections
