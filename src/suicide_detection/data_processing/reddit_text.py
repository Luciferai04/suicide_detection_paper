import re
from typing import Callable

# Basic normalization: remove Reddit mentions /u/ and subreddits /r/, markdown links, normalize elongations
USER_MENTION = re.compile(r"(?i)\b/u/\w+")
SUBREDDIT = re.compile(r"(?i)\b/r/\w+")
MARKDOWN_LINK = re.compile(r"\[(.*?)\]\((.*?)\)")
URL = re.compile(r"https?://\S+|www\.[^\s]+", re.IGNORECASE)
ELONG = re.compile(r"(\w)\1{2,}")  # soooo -> soo -> so


def _normalize_elongations(text: str) -> str:
    def repl(m):
        return m.group(1) * 2
    return ELONG.sub(repl, text)


def reddit_clean(text: str) -> str:
    s = text
    s = MARKDOWN_LINK.sub(lambda m: m.group(1), s)
    s = USER_MENTION.sub("<USER>", s)
    s = SUBREDDIT.sub("<SUBREDDIT>", s)
    s = URL.sub("<URL>", s)
    s = _normalize_elongations(s)
    # Collapse excessive whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
