import pytest

from graphrag_tagger.lda.sk_modelling import SklearnTopicExtractor


def test_fit_with_empty_texts():
    extractor = SklearnTopicExtractor()
    with pytest.raises(ValueError):
        extractor.fit([])


def test_get_topics():
    texts = [
        "this is a test document",
        "another test document",
        "yet another test document",
    ]
    extractor = SklearnTopicExtractor(n_components=2, max_features=20)
    extractor.fit(texts)
    topics = extractor.get_topics()
    assert isinstance(topics, list)
    assert all(isinstance(topic, str) for topic in topics)


def test_transform():
    texts = [
        "this is a test document",
        "another test document",
        "yet another test document",
    ]
    extractor = SklearnTopicExtractor(n_components=2, max_features=20)
    extractor.fit(texts)
    topic_distribution = extractor.transform(texts)
    assert len(topic_distribution) == len(texts)
    for distrib in topic_distribution:
        assert len(distrib) == extractor.n_components
