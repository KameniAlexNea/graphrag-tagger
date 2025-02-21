import os
import tempfile
import unittest

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


class TestTagger(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory with a dummy pdf file.
        self.test_dir = tempfile.TemporaryDirectory()
        self.pdf_path = os.path.join(self.test_dir.name, "test.pdf")
        with open(self.pdf_path, "w") as f:
            f.write("dummy pdf content")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_pdf_texts_returns_text(self):
        # Monkey-patch os.listdir and fitz.open.
        original_listdir = os.listdir
        import fitz

        original_fitz_open = fitz.open
        os.listdir = lambda folder: ["test.pdf", "ignore.txt"]
        fitz.open = dummy_fitz_open

        texts = load_pdf_texts(self.test_dir.name)
        self.assertTrue(isinstance(texts, dict))
        self.assertIn(self.pdf_path, texts)
        self.assertIn("dummy text", texts[self.pdf_path])

        # Restore original functions.
        os.listdir = original_listdir
        fitz.open = original_fitz_open


if __name__ == "__main__":
    unittest.main()
