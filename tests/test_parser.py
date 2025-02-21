import unittest

from graphrag_tagger.chat.parser import parse_json


class TestParser(unittest.TestCase):
    def test_parse_valid_json(self):
        json_str = '{"key": "value"}'
        result = parse_json(json_str)
        self.assertEqual(result, {"key": "value"})

    def test_parse_json_with_markers(self):
        json_str = """
        Some text before.
        ```json
        {"tags": ["a", "b"]}
        ```
        Some text after.
        """
        result = parse_json(json_str)
        self.assertEqual(result, {"tags": ["a", "b"]})

    def test_parse_fallback_curly(self):
        json_str = 'Prefix text {"key": "value"} suffix'
        result = parse_json(json_str)
        self.assertEqual(result, {"key": "value"})

    def test_parse_fallback_square(self):
        json_str = "Some text [1, 2, 3] more text"
        result = parse_json(json_str)
        self.assertEqual(result, [1, 2, 3])

    def test_parse_invalid(self):
        json_str = "Not a json string"
        result = parse_json(json_str)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
