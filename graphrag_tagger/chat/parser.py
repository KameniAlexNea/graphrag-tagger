import json
import re


def parse_json(json_str: str):
    """
    Parses a JSON object from a string that may contain extra text.
    
    This function attempts three approaches to extract JSON:
    
    1. Directly parsing the entire string.
    2. Extracting JSON enclosed within triple backticks (```json ... ```).
    3. Extracting content between the first '{' and the last '}' or between '[' and ']'.
    
    :param json_str: The input string potentially containing a JSON object.
    :type json_str: str
    :return: The parsed JSON object if successfully extracted, otherwise None.
    :rtype: dict or list or None
    """
    # Attempt 1: Try to load the entire string as JSON.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Look for a JSON block delimited by ```json and ```.
    match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
    if match:
        json_block = match.group(1)
        try:
            return json.loads(json_block)
        except json.JSONDecodeError as e:
            print("Error parsing JSON block:", e)
            return None

    # Attempt 3: Fallback to extracting text between the first '{' and the last '}'.
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_block = json_str[start : end + 1]
        try:
            return json.loads(json_block)
        except json.JSONDecodeError as e:
            print("Error parsing fallback JSON block:", e)
            return None

    # Attempt 3: Fallback to extracting text between the first '{' and the last '}'.
    start = json_str.find("[")
    end = json_str.rfind("]")
    if start != -1 and end != -1 and end > start:
        json_block = json_str[start : end + 1]
        try:
            return json.loads(json_block)
        except json.JSONDecodeError as e:
            print("Error parsing fallback JSON block:", e)
            return None

    # If no JSON could be parsed, return None.
    return None


# --- Example Usage ---
if __name__ == "__main__":
    # Example input with extra text and JSON delimited by ```json markers.
    example_text = """
    Here is some verbose text.
    ```json
    {
        "tags": ["technology", "health", "finance"]
    }
    ```
    Some more irrelevant text here.
    """
    parsed = parse_json(example_text)
    print("Parsed JSON:", parsed)
