from graphrag_tagger.utilities.parser import parse_json


def test_parse_valid_json():
    json_str = '{"key": "value"}'
    result = parse_json(json_str)
    assert result == {"key": "value"}


def test_parse_json_with_markers():
    json_str = """
    Some text before.
    ```json
    {"tags": ["a", "b"]}
    ```
    Some text after.
    """
    result = parse_json(json_str)
    assert result == {"tags": ["a", "b"]}


def test_parse_fallback_curly():
    json_str = 'Prefix text {"key": "value"} suffix'
    result = parse_json(json_str)
    assert result == {"key": "value"}


def test_parse_fallback_square():
    json_str = "Some text [1, 2, 3] more text"
    result = parse_json(json_str)
    assert result == [1, 2, 3]


def test_parse_invalid():
    json_str = "Not a json string"
    result = parse_json(json_str)
    assert result is None
