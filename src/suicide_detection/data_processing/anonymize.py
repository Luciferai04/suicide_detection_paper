import re
from dataclasses import dataclass
from typing import Optional

try:
    import spacy
    from spacy.matcher import Matcher

    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


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
    PHONE_RE: re.Pattern = re.compile(
        r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b"
    )

    def _spacy_mask(self, text: str) -> Optional[str]:
        import os

        # Fast path: allow disabling spaCy NER masking via env var for large-batch preprocessing
        if os.environ.get("ANON_FAST", "0") == "1":
            return None
        if not _SPACY_AVAILABLE:
            return None
        try:
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = spacy.blank("en")
            matcher = Matcher(nlp.vocab)
            matcher.add(
                "DATE",
                [
                    [
                        {"SHAPE": "dd"},
                        {"TEXT": "/"},
                        {"SHAPE": "dd"},
                        {"TEXT": "/"},
                        {"SHAPE": "dddd"},
                    ]
                ],
            )
            doc = nlp(text)
            s = text
            for ent in getattr(doc, "ents", []):
                if ent.label_ in {"PERSON", "GPE", "ORG", "LOC"}:
                    s = s.replace(ent.text, "<PII>")
            for match_id, start, end in matcher(doc):
                span = doc[start:end]
                s = s.replace(span.text, "<DATE>")
            return s
        except Exception:
            return None
            return None

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
        masked = self._spacy_mask(s)
        return masked if masked is not None else s
