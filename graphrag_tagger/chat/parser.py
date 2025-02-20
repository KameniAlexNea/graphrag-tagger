import re
import json

def parse_json(json_str: str):
    """
    Attempt to parse a JSON object from a string that may include extra verbose text.
    This function first tries to parse the whole string. If that fails, it looks for a block
    delimited by ```json and ```. If still unsuccessful, it falls back to extracting the substring
    between the first '{' and the last '}'.
    
    Parameters:
        json_str (str): The input string that may contain a JSON block along with extra text.
    
    Returns:
        The parsed JSON object, or None if no valid JSON could be found.
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
        json_block = json_str[start:end+1]
        try:
            return json.loads(json_block)
        except json.JSONDecodeError as e:
            print("Error parsing fallback JSON block:", e)
            return None
        
    # Attempt 3: Fallback to extracting text between the first '{' and the last '}'.
    start = json_str.find("[")
    end = json_str.rfind("]")
    if start != -1 and end != -1 and end > start:
        json_block = json_str[start:end+1]
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
