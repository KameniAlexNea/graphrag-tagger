from graphrag_tagger.chat.llm import LLM, LLMService


# Create a DummyLLM that overrides __call__.
class DummyLLMService(LLMService):
    def __init__(self, model="test-model"):
        # Skip actual initialization.
        self.model_name = model

    def __call__(self, messages: list):
        prompt = messages[0]["content"]
        if "transform a list of messy topics" in prompt:
            return '{"topics": ["TopicA", "TopicB"]}'
        elif "analyze a given text excerpt" in prompt:
            return '["TopicA", "TopicC"]'
        return "{}"


class DummyLLM(LLM):
    def __init__(self, llm_service):
        self.llm_service = llm_service


def test_clean_topics():
    llm_service = DummyLLMService()
    llm = DummyLLM(llm_service)
    result = llm.clean_topics(["messy topic one", "messy topic two"])
    assert result == {"topics": ["TopicA", "TopicB"]}


def test_classify():
    llm_service = DummyLLMService()
    llm = DummyLLM(llm_service)
    result = llm.classify("Some document text", ["TopicA", "TopicB", "TopicC"])
    assert result == ["TopicA", "TopicC"]
