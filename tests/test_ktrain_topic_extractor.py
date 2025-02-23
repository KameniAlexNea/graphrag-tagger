import pytest

# Patch get_topic_model in kt_modelling.
import graphrag_tagger.lda.kt_modelling as kt_modelling
from graphrag_tagger.lda.kt_modelling import KtrainTopicExtractor


# Dummy topic model to simulate ktrain behavior.
class DummyTopicModel:
    def __init__(self, texts, n_features, min_df, max_df, n_topics):
        self.texts = texts
        self.n_topics = n_topics if n_topics is not None else 2

    def build(self, texts, threshold):
        pass

    def get_topics(self):
        return ["topic1", "topic2"]

    def filter(self, texts):
        return texts

    def predict(self, texts, threshold):
        return [[0.1] * self.n_topics for _ in texts]


def dummy_get_topic_model(texts, n_features, min_df, max_df, n_topics):
    return DummyTopicModel(texts, n_features, min_df, max_df, n_topics)


kt_modelling.get_topic_model = dummy_get_topic_model


def test_fit_with_empty_texts():
    extractor = KtrainTopicExtractor()
    with pytest.raises(ValueError):
        extractor.fit([])


def test_get_topics():
    texts = ["document one", "document two", "document three", "document four"]
    extractor = KtrainTopicExtractor(
        n_components=2, n_features=3, min_df=0.0, max_df=1.0
    )
    extractor.fit(texts)
    topics = extractor.get_topics()
    assert topics == ["topic1", "topic2"]
