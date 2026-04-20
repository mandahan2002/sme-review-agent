"""Tools for the SME Review Agent."""
from .m365_search import m365_configured, search_m365
from .web_search import search_web

__all__ = ["search_web", "search_m365", "m365_configured"]
