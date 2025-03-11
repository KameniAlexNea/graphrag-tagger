import json

import pytest

from graphrag_tagger.tagger import main


# Dummy PDF classes to simulate fitz behavior.
class DummyPage:
    def get_text(self):
        return """
focused summarization (QFS) task, rather than an explicit retrieval task. Prior
QFS methods, meanwhile, do not scale to the quantities of text indexed by typ-
ical RAG systems. To combine the strengths of these contrasting methods, we
propose GraphRAG, a graph-based approach to question answering over private
text corpora that scales with both the generality of user questions and the quantity
of source text. Our approach uses an LLM to build a graph index in two stages:
first, to derive an entity knowledge graph from the source documents, then to pre-
generate community summaries for all groups of closely related entities. Given a
question, each community summary is used to generate a partial response, before
all partial responses are again summarized in a final response to the user. For a
class of global sensemaking questions over datasets in the 1 million token range,
we show that GraphRAG leads to substantial improvements over a conventional
RAG baseline for both the comprehensiveness and diversity of generated answers.
1
Introduction
Retrieval augmented generation (RAG) (Lewis et al., 2020) is an established approach to using
"""


class DummyDoc:
    def __init__(self, text):
        self.text = text
        self.pages = [DummyPage()]

    def __iter__(self):
        return iter(self.pages)


def dummy_fitz_open(file_path):
    return DummyDoc("Dummy PDF content.")


# Dummy topic extractor ignoring input for testing purposes.
class DummyTopicExtractor:
    def fit(self, texts):
        return self

    def get_topics(self):
        return ["dummy_topic1", "dummy_topic2"]

    def transform(self, texts):
        return [[0.5, 0.5] for _ in texts]

    def filter_texts(self, texts):
        return texts


# Dummy LLM that returns fixed responses.
class DummyLLM:
    def __init__(self, model="dummy-model"):
        self.model_name = model

    def __call__(self, messages: list):
        prompt = messages[0]["content"]
        if "transform a list of messy topics" in prompt:
            return '{"topics": ["clean_topic1", "clean_topic2"]}'
        elif "analyze a given text excerpt" in prompt:
            return '["clean_topic1"]'
        return "{}"

    def clean_topics(self, topics: list):
        return {"topics": ["clean_topic1", "clean_topic2"]}

    def classify(self, document_chunk: str, topics: list):
        return ["clean_topic1"]


@pytest.fixture(autouse=True)
def override_dependencies(monkeypatch):
    # Override fitz.open used in load_pdf_texts.
    import fitz

    monkeypatch.setattr(fitz, "open", dummy_fitz_open)
    # Override both topic extractors in the pipeline with DummyTopicExtractor.
    dummy_extractor = DummyTopicExtractor()
    monkeypatch.setattr(
        "graphrag_tagger.tagger.SklearnTopicExtractor", lambda **kwargs: dummy_extractor
    )
    monkeypatch.setattr(
        "graphrag_tagger.tagger.KtrainTopicExtractor", lambda **kwargs: dummy_extractor
    )
    # Override LLM in tagger.py.
    monkeypatch.setattr("graphrag_tagger.tagger.LLM", DummyLLM)


def test_main_pipeline(tmp_path):
    # Create a temporary PDF folder with a dummy PDF file.
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_file = pdf_dir / "dummy.pdf"
    pdf_file.write_text(
        """
focused summarization (QFS) task, rather than an explicit retrieval task. Prior
QFS methods, meanwhile, do not scale to the quantities of text indexed by typ-
ical RAG systems. To combine the strengths of these contrasting methods, we
propose GraphRAG, a graph-based approach to question answering over private
text corpora that scales with both the generality of user questions and the quantity
of source text. Our approach uses an LLM to build a graph index in two stages:
first, to derive an entity knowledge graph from the source documents, then to pre-
generate community summaries for all groups of closely related entities. Given a
question, each community summary is used to generate a partial response, before
all partial responses are again summarized in a final response to the user. For a
class of global sensemaking questions over datasets in the 1 million token range,
we show that GraphRAG leads to substantial improvements over a conventional
RAG baseline for both the comprehensiveness and diversity of generated answers.
1
Introduction
Retrieval augmented generation (RAG) (Lewis et al., 2020) is an established approach to using
"""
    )
    # Create an output folder.
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Prepare parameters for the main pipeline.
    params = {
        "pdf_folder": str(pdf_dir),
        "chunk_size": 50,
        "chunk_overlap": 0,
        "n_components": 2,
        "n_features": 100,
        "min_df": 2,
        "max_df": 0.95,
        "llm_model": "dummy-model",
        "output_folder": str(output_dir),
        "model_choice": "sk",
    }
    # Run the pipeline.
    main(params)

    # Verify that JSON chunk files were created.
    output_files = list(output_dir.glob("chunk_*.json"))
    assert len(output_files) > 0, "No output chunk files were created."

    # Validate the content of one JSON file.
    sample_file = output_files[0]
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "chunk" in data
    assert "source_file" in data
    assert "classification" in data
