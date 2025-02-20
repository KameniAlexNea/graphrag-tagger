from typing import List
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class SklearnTopicExtractor:
    def __init__(
        self,
        texts: List[str],
        n_components: int = None,
        max_features: int = 512,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """
        Initialize the topic extractor using scikit-learn's LDA.

        Parameters:
            texts: List of strings, where each string is a document.
            n_components: Number of topics to extract.
            max_features: Maximum number of features for the CountVectorizer.
            min_df: Minimum document frequency for CountVectorizer.
            max_df: Maximum document frequency for CountVectorizer.
            n_top_words: Number of top words to display per topic.
        """
        if n_components is None:
            n_components = min(int(len(texts) ** 0.5), 25)
            print("n_components is None, setting it to sqrt(len(texts)):", n_components)
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.texts = texts

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
            n_top_words: Number of top words to display per topic.

        Returns:
            self (for chaining)
        """
        X = self.vectorizer.fit_transform(texts)
        self.lda.fit(X)
        return self

    def get_topics(self, threshold_fraction: float = 0.8, n_word_limit = 15) -> List[str]:
        """
        Retrieve topics as a list of strings. Each topic is represented as a space-separated
        string of words that together account for at least `threshold_fraction` of the total topic weight.
        
        Parameters:
            threshold_fraction (float): Fraction of the total weight to cover (default is 0.8, i.e. 80%).
            
        Returns:
            List[str]: A list of topic strings.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic in self.lda.components_:
            total_weight = topic.sum()
            cumulative_weight = 0.0
            # Sort indices in descending order (highest weight first)
            sorted_indices = topic.argsort()[::-1]
            selected_indices = []
            pos = 0
            
            for idx in sorted_indices:
                pos += 1
                selected_indices.append(idx)
                cumulative_weight += topic[idx]
                if cumulative_weight >= threshold_fraction * total_weight or pos >= n_word_limit:
                    break
            
            top_words = [feature_names[i] for i in selected_indices]
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
    from sklearn.datasets import fetch_20newsgroups

    # Sample texts – in practice, these would be your document texts.
    remove = ("headers", "footers", "quotes")
    # newsgroups_train = fetch_20newsgroups(subset="train", remove=remove)
    newsgroups_test = fetch_20newsgroups(subset="test", remove=remove)
    texts: list[str] =  newsgroups_test.data # + newsgroups_train.data

    # Initialize and fit the topic extractor.
    extractor = SklearnTopicExtractor(texts)
    extractor.fit(texts)

    # Retrieve the extracted (messy) topics.
    messy_topics = extractor.get_topics(0.8)
    print("Topics from LDA:")
    for topic in messy_topics:
        print("-", topic)

    # Optionally, transform new texts into the topic space.
    topic_distributions = extractor.transform(texts[-10:])
    print("\nTopic distributions for sample texts:")
    print(topic_distributions)
