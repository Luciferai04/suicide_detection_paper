import re
from dataclasses import dataclass
from typing import List


@dataclass
class Anonymizer:
    """Anonymize PII from social media text while preserving clinically relevant cues.

    This removes URLs, emails, phone numbers, @mentions, and replaces them with
    placeholders. Emojis, emphasis, and punctuation are preserved.
    """

    replace_usernames: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    replace_phones: bool = True

    USER_RE: re.Pattern = re.compile(r"@\w+")
    URL_RE: re.Pattern = re.compile(r"https?://\S+|www\.[^\s]+", re.IGNORECASE)
    EMAIL_RE: re.Pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE: re.Pattern = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b")

    def transform(self, text: str) -> str:
        s = text
        if self.replace_urls:
            s = self.URL_RE.sub("<URL>", s)
        if self.replace_emails:
            s = self.EMAIL_RE.sub("<EMAIL>", s)
        if self.replace_phones:
            s = self.PHONE_RE.sub("<PHONE>", s)
        if self.replace_usernames:
            s = self.USER_RE.sub("<USER>", s)
        return s

