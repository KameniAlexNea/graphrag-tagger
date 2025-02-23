from typing import List, Optional

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class SklearnTopicExtractor:
    """
    Topic extractor using scikit-learn's Latent Dirichlet Allocation (LDA).
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        n_features: int = 512,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """
        Initializes the topic extractor.

        :param n_components: Number of topics to extract. If None, it is set to the square root of the number of
                             documents used during fitting, capped at 25. Defaults to None.
        :type n_components: Optional[int]
        :param n_features: Maximum number of features (words) to consider in the vocabulary.
        :type n_features: int
        :param min_df: Minimum document frequency for a word to be included in the vocabulary.
        :type min_df: int
        :param max_df: Maximum document frequency for a word to be included in the vocabulary.
        :type max_df: float
        """
        self.n_components = n_components
        self.max_features = n_features
        self.min_df = min_df
        self.max_df = max_df

        # Initialize CountVectorizer and LDA model (will be fitted later)
        self.vectorizer: Optional[CountVectorizer] = None
        self.lda: Optional[LatentDirichletAllocation] = None

    def fit(self, texts: List[str]):
        """
        Fits the LDA topic model on a list of document texts.

        :param texts: List of strings, where each string is a document.
        :type texts: List[str]
        :raises ValueError: If the input list is empty.
        :return: Returns the instance itself to allow for method chaining.
        :rtype: SklearnTopicExtractor
        """
        if not texts:
            raise ValueError("Input 'texts' list cannot be empty.")

        if self.n_components is None:
            self.n_components = min(int(len(texts) ** 0.5), 25)
            print(
                f"n_components is None, setting it to sqrt(len(texts)): {self.n_components}"
            )

        self.vectorizer = CountVectorizer(
            max_features=self.max_features, min_df=self.min_df, max_df=self.max_df
        )
        self.lda = LatentDirichletAllocation(
            n_components=self.n_components, random_state=0
        )

        # Fit CountVectorizer to create document-term matrix
        X = self.vectorizer.fit_transform(texts)
        # Fit LDA model
        self.lda.fit(X)
        return self

    def get_topics(
        self, threshold_fraction: float = 0.8, n_word_limit: int = 10
    ) -> List[str]:
        """
        Retrieves topics as a list of strings.

        Each topic is represented as a space-separated string of words that together
        account for at least `threshold_fraction` of the total topic weight,
        or up to `n_word_limit` words, whichever comes first.

        :param threshold_fraction: Fraction of the total weight to cover (default is 0.8, i.e., 80%).
        :type threshold_fraction: float
        :param n_word_limit: Maximum number of words to include in each topic string.
        :type n_word_limit: int
        :raises ValueError: If the model has not been fitted yet.
        :return: A list of topic strings.
        :rtype: List[str]
        """
        if self.vectorizer is None or self.lda is None:
            raise ValueError(
                "The model must be fitted first. Call 'fit' method before 'get_topics'."
            )

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_weights in self.lda.components_:
            total_weight = topic_weights.sum()
            cumulative_weight = 0.0
            selected_word_indices: List[int] = []
            sorted_indices = topic_weights.argsort()[
                ::-1
            ]  # Descending order of weights
            word_count = 0

            for word_index in sorted_indices:
                word_count += 1
                selected_word_indices.append(word_index)
                cumulative_weight += topic_weights[word_index]
                if (
                    cumulative_weight >= threshold_fraction * total_weight
                    or word_count >= n_word_limit
                ):
                    break

            top_words = [feature_names[i] for i in selected_word_indices]
            topics.append(" ".join(top_words))

        return topics


# ----- Example usage -----
if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups

    # Sample texts from 20 newsgroups dataset
    remove_categories = ("headers", "footers", "quotes")
    newsgroups_test = fetch_20newsgroups(subset="test", remove=remove_categories)
    texts: List[str] = newsgroups_test.data

    # Initialize and fit the topic extractor.
    extractor = SklearnTopicExtractor()
    extractor.fit(texts)

    # Retrieve the extracted topics.
    extracted_topics = extractor.get_topics(0.8)
    print("Extracted Topics from LDA:")
    for topic in extracted_topics:
        print("-", topic)
