import re
import string

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextCleaner:
    def __init__(self, chunk_size=512, chunk_overlap=75):
        encoding = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(encoding.encode(x)),
        )

    def split_text(self, text):
        texts = self.text_splitter.split_text(text)
        print(texts)
        texts = [self.merge_sentences(i) for i in texts if self.is_valid_file(i)]
        return texts

    @staticmethod
    def merge_sentences(text: str) -> str:
        """
        Merge lines from OCR/PDF text by handling common token splits.

        Rules:
            - If a line ends with a hyphen, remove the hyphen and merge it with the next line.
            - If the previous (merged) line ends with punctuation (".", "?", "!"),
                start a new line with the current text.
            - If the current line is all uppercase (likely a title/heading),
                start a new line.
            - If the current line starts with an uppercase letter and the previous line
                does not end with punctuation, assume a new sentence/paragraph.
            - Otherwise, merge the current line with a space.

            Parameters:
                text (str): The input text with potential unwanted line breaks.

        Returns:
            str: The cleaned and merged text.
        """
        punctuation = {".", "?", "!"}
        # Split text into non-empty, stripped lines.
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return ""

        merged = [lines[0]]
        for line in lines[1:]:
            last_line = merged[-1]
            # If the last line ends with a hyphen, remove it and merge without a space.
            if last_line.endswith("-"):
                merged[-1] = last_line[:-1] + line
            # If the last line ends with sentence punctuation, start a new line.
            elif last_line and last_line[-1] in punctuation:
                merged.append(line)
            # If the current line is all uppercase, treat it as a title or heading.
            elif line.isupper():
                merged.append(line)
            # If the current line starts with uppercase and the previous line doesn't end with punctuation,
            # assume it's a new sentence/paragraph.
            elif (
                line
                and line[0].isupper()
                and (not last_line or last_line[-1] not in punctuation)
            ):
                merged.append(line)
            # Otherwise, join the current line with a space.
            else:
                merged[-1] = last_line + " " + line

        return "\n".join(merged).strip()

    @staticmethod
    def _is_potential_title(line: str) -> bool:
        line = line.strip()  # Remove leading and trailing whitespace

        # Check if the line starts with a number followed by a period (e.g., "19.")
        if re.match(r"^\d+\.", line):
            return True

        # Other heuristics to identify if it's likely a title
        if len(line.split()) <= 10 and not line.endswith(
            tuple(string.punctuation)
        ):  # Short and no ending punctuation
            if line[0].isupper():  # Starts with an uppercase letter
                words = line.split()
                # Optional: Check for capitalized words in the title
                capitalized_words = sum(1 for word in words if word[0].isupper())
                # Allow a few lowercase words if there are capitalized ones
                if capitalized_words >= len(words) / 2:
                    return True

        return False

    @staticmethod
    def is_valid_file(raw: str):
        lines = [i for i in raw.split("\n") if i.strip()]
        count = [i for i in lines if "........" in i]  # potential title
        count2 = [i for i in lines if "—." in i]  # potential citation
        count3 = [
            1 for i in lines if TextCleaner._is_potential_title(i)
        ]  # potential citation part
        return (
            (len(lines) > 3)
            and len(count) < 5
            and len(count2) < 5
            and len(count3) < 5
        )

    @staticmethod
    def split_text_into_large_chunks(text: str, target_word_count=300):
        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        chunks: list[str] = []
        current_chunk = []
        current_word_count = 0
        for paragraph in paragraphs:
            word_count = len(re.findall(r"\w+", paragraph))
            if current_word_count + word_count >= target_word_count:
                current_chunk.append(paragraph)
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            current_chunk.append(paragraph)
            current_word_count += word_count

        # Ajouter le dernier chunk s'il reste des paragraphes
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks
