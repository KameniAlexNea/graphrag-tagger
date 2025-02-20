from typing import List
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class SklearnTopicExtractor:
    def __init__(
        self,
        n_components: int = 5,
        max_features: int = 512,
        min_df: int = 2,
        max_df: float = 0.95,
        n_top_words: int = 10,
    ):
        """
        Initialize the topic extractor using scikit-learn's LDA.

        Parameters:
            n_components: Number of topics to extract.
            max_features: Maximum number of features for the CountVectorizer.
            min_df: Minimum document frequency for CountVectorizer.
            max_df: Maximum document frequency for CountVectorizer.
            n_top_words: Number of top words to display per topic.
        """
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.n_top_words = n_top_words

        # Create a CountVectorizer to produce a token count matrix.
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, min_df=self.min_df, max_df=self.max_df
        )
        # Create the LDA model.
        self.lda = LatentDirichletAllocation(
            n_components=self.n_components, random_state=0
        )

    def fit(self, texts: List[str]):
        """
        Fit the topic model on a list of document texts.

        Parameters:
            texts: List of strings, where each string is a document.

        Returns:
            self (for chaining)
        """
        X = self.vectorizer.fit_transform(texts)
        self.lda.fit(X)
        return self

    def get_topics(self, n_top_words=None) -> List[str]:
        """
        Retrieve topics as a list of strings. Each topic is represented as a space-separated
        string of the top words.

        Returns:
            List of topic strings.
        """
        if n_top_words is None:
            n_top_words = self.n_top_words
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            # Get indices of the top words for this topic.
            top_indices = topic.argsort()[: -self.n_top_words - 1 : -1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append(" ".join(top_words))
        return topics

    def transform(self, texts: List[str]):
        """
        Transform new texts into the topic distribution space.

        Parameters:
            texts: List of document texts.

        Returns:
            Array of shape (n_samples, n_components) with topic probabilities.
        """
        X = self.vectorizer.transform(texts)
        return self.lda.transform(X)


# ----- Example usage -----
if __name__ == "__main__":
    # Sample texts – in practice, these would be your document texts.
    texts = [
        "The stock market crashed due to high inflation and economic uncertainty.",
        "Advancements in machine learning and artificial intelligence are transforming technology.",
        "The local football team won the championship after a thrilling season.",
        "Healthcare innovations are improving patient outcomes and reducing costs.",
    ]

    # Initialize and fit the topic extractor.
    extractor = SklearnTopicExtractor(n_components=5, n_top_words=5)
    extractor.fit(texts)

    # Retrieve the extracted (messy) topics.
    messy_topics = extractor.get_topics()
    print("Messy Topics from LDA:")
    for topic in messy_topics:
        print("-", topic)

    # Optionally, transform new texts into the topic space.
    topic_distributions = extractor.transform(texts)
    print("\nTopic distributions for sample texts:")
    print(topic_distributions)
