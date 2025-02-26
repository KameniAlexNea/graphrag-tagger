CREATE_TOPICS = """
You are an expert in summarizing and clarifying topics. Your task is to transform a list of messy topics generated by an LDA model into clear, concise topic labels.

Here is the list of messy topics:
<topics>
{topics}
</topics>

Follow these steps to complete the task:

1. Read through each messy topic carefully.
2. Identify the main theme or concept represented by the words in each topic.
3. Create a clear, concise label that accurately represents the essence of each topic.
4. Ensure that each label is brief (preferably 1-5 words comma separated) but descriptive enough to distinguish it from other topics.
5. Avoid using vague or overly general terms.

Present your answer as a JSON list of strings. Do not include any extra text or explanations outside of the JSON list.

Example of expected output format:
["Technology", "Environmental Issues", "Sports", "Politics", "Education"]

Remember to focus on clarity and conciseness in your topic labels. Each label should effectively capture the main idea of the original messy topic."""


CLASSIFY_PROMPT = """
You are an expert content classifier. Your task is to analyze a given text excerpt and perform the following:

1. **Determine the type of the content.**
2. **Assess if the content is sufficient for topic classification.**
3. **If sufficient, select up to 3 topics from the provided list that best describe the content.**

Here is the text excerpt you need to analyze:
```
<text>
{text}
</text>
```

Here are the candidate topics to choose from:
```
<topics>
{topics}
</topics>
```

### Instructions:
1. **Determine the content type** from the following list: 
   - paragraph
   - table
   - list
   - header
   - footer
   - index
   - figure_caption
   - other
2. **Assess sufficiency**: Evaluate if the text excerpt contains sufficient information to reliably determine its main themes or subject matter. If it does, set `"is_sufficient"` to `true`; otherwise, set it to `false`.
3. **Select topics**: If `"is_sufficient"` is `true`, select up to 3 topics from the provided list that best describe the content. If fewer than 3 topics are relevant, select fewer. If `"is_sufficient"` is `false`, do not select any topics.

### Output Format:
Output your answer as a JSON object with the following keys:
- `"content_type"`: A string indicating the type of content (from the list above).
- `"is_sufficient"`: A boolean indicating if the content is sufficient for topic classification.
- `"topics"`: A JSON array of strings containing the selected topics (empty if `"is_sufficient"` is `false`).

#### Example Outputs:
```json
{
  "content_type": "paragraph",
  "is_sufficient": true,
  "topics": ["Topic1", "Topic2"]
}
```
```json
{
  "content_type": "footer",
  "is_sufficient": false,
  "topics": []
}
```

### Reminders:
- Select no more than 3 topics.
- Use only topics from the provided list.
- Ensure the selected topics are the most relevant to the text content if `"is_sufficient"` is `true`.

Present your answer as a JSON object. Do not include any extra text or explanations outside of the JSON object.
""".strip()
