import os

from graphrag_tagger.tagger import load_pdf_texts


# Dummy classes to simulate a PDF document using fitz.
class DummyPage:
    def get_text(self):
        return "dummy text"


class DummyDoc:
    def __init__(self, text):
        self.text = text
        self.pages = [DummyPage()]

    def __iter__(self):
        return iter(self.pages)


def dummy_fitz_open(file_path):
    return DummyDoc("dummy text")


def test_load_pdf_texts_returns_text(tmp_path, monkeypatch):
    # Create a temporary directory with a dummy pdf file.
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("dummy pdf content")
    # Monkey-patch os.listdir and fitz.open.
    monkeypatch.setattr(os, "listdir", lambda folder: ["test.pdf", "ignore.txt"])
    import fitz

    monkeypatch.setattr(fitz, "open", dummy_fitz_open)
    texts = load_pdf_texts(str(tmp_path))
    expected_path = os.path.join(str(tmp_path), "test.pdf")
    assert isinstance(texts, dict)
    assert expected_path in texts
    assert "dummy text" in texts[expected_path]
