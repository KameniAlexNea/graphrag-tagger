import unittest

from graphrag_tagger.lda.sk_modelling import SklearnTopicExtractor


class TestSklearnTopicExtractor(unittest.TestCase):
    def test_fit_with_empty_texts(self):
        extractor = SklearnTopicExtractor()
        with self.assertRaises(ValueError):
            extractor.fit([])

    def test_get_topics(self):
        texts = [
            "this is a test document", 
            "another test document",
            "yet another test document"
        ]
        extractor = SklearnTopicExtractor(n_components=2, max_features=20)
        extractor.fit(texts)
        topics = extractor.get_topics()
        self.assertIsInstance(topics, list)
        self.assertTrue(all(isinstance(topic, str) for topic in topics))

    def test_transform(self):
        texts = [
            "this is a test document", 
            "another test document",
            "yet another test document"
        ]
        extractor = SklearnTopicExtractor(n_components=2, max_features=20)
        extractor.fit(texts)
        topic_distribution = extractor.transform(texts)
        self.assertEqual(len(topic_distribution), len(texts))
        for distrib in topic_distribution:
            self.assertEqual(len(distrib), extractor.n_components)


if __name__ == "__main__":
    unittest.main()
