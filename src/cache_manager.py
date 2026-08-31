"""
cache_manager.py
----------------
Demonstrates LangChain's two built-in LLM caches:

- InMemoryCache: lives in RAM, cleared when the app restarts, fastest.
- SQLiteCache: stored in a .db file on disk, survives restarts, still fast.

Call set_cache_mode() once when the sidebar toggle changes -- LangChain
then checks the active cache automatically before every model call, so an
identical request the second time round returns instantly without hitting
the OpenAI API again.
"""

try:
    from langchain.globals import set_llm_cache
except ModuleNotFoundError:
    from langchain_core.globals import set_llm_cache

from langchain_community.cache import InMemoryCache, SQLiteCache

SQLITE_CACHE_PATH = ".langchain_cache.db"


def set_cache_mode(mode: str) -> str:
    """
    mode is one of the strings in config.CACHE_MODES.
    Returns a short human-readable status message for the UI.
    """
    if mode.startswith("In-Memory"):
        set_llm_cache(InMemoryCache())
        return "In-memory cache active -- cleared when the app restarts."
    if mode.startswith("SQLite"):
        set_llm_cache(SQLiteCache(database_path=SQLITE_CACHE_PATH))
        return f"SQLite cache active -- stored in {SQLITE_CACHE_PATH}, survives restarts."
    set_llm_cache(None)
    return "Caching disabled -- every submission calls the API fresh."
