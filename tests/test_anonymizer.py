from pathlib import Path

from suicide_detection.data_processing.anonymize import Anonymizer


def test_anonymizer_masks_pii():
    anon = Anonymizer()
    text = "Contact me at user@example.com or visit https://example.com. Call +1-415-555-1234 and ping @john."
    out = anon.transform(text)
    assert "<EMAIL>" in out
    assert "<URL>" in out
    assert "<PHONE>" in out
    assert "<USER>" in out


def test_anonymizer_preserves_emphasis():
    anon = Anonymizer()
    text = "I am fine!!! really???"
    out = anon.transform(text)
    assert "!!!" in out and "???" in out
