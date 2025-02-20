import aisuite as ai

from .parser import parse_json
from .prompts import CREATE_TOPICS


class LLM:
    def __init__(self, model="ollama:phi4"):
        self.model = ai.Client()
        self.model_name = model

    def __call__(self, messages: list):
        return (
            self.model.chat.completions.create(
                model=self.model_name, temperature=0.75, messages=messages
            )
            .choices[0]
            .message.content
        )

    def clean_topics(self, topics: list):
        topics_str = "\n".join(topics)
        prompt = CREATE_TOPICS.format(topics=topics_str)
        results = self.__call__([{"role": "system", "content": prompt}])
        return parse_json(results)

    def classify(self, document_chunk: str, topics: list):
        topics_str = "\n".join(topics)
        prompt = CREATE_TOPICS.format(text=document_chunk, topics=topics_str)
        results = self.__call__([{"role": "system", "content": prompt}])
        return parse_json(results)
