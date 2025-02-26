import argparse
import json
import os

import fitz
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .chat.llm import LLM
from .lda.kt_modelling import KtrainTopicExtractor
from .lda.sk_modelling import SklearnTopicExtractor


def load_pdf_texts(folder_path: str):
    """
    Loads text from all PDFs in a given folder.

    :param folder_path: Path to the folder containing PDF files.
    :type folder_path: str
    :return: Dictionary where keys are file paths and values are extracted text.
    :rtype: dict
    """
    texts = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            doc = fitz.open(file_path)
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text() + "\n"
            texts[file_path] = pdf_text
    return texts


def main(params: dict):
    """
    Main function to extract text from PDFs, perform topic modeling, clean topics using LLM, and save results.

    :param params: Dictionary containing processing parameters.
    :type params: dict
    """
    # Load PDFs from folder
    pdf_texts = load_pdf_texts(params["pdf_folder"])

    # Split texts into chunks along with source file metadata
    all_chunks = []  # each element is dict: {"chunk": str, "source_file": str}

    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        length_function=lambda x: len(encoding.encode(x)),
    )

    # Split texts into chunks along with source file metadata
    for file_path, text in pdf_texts.items():
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({"chunk": chunk, "source_file": file_path})

    if not all_chunks:
        print("No texts extracted from PDFs.")
        return

    print(f"Total chunk texts: {len(all_chunks)}")

    # Choose the topic extractor based on model_choice parameter
    topic_class = (
        SklearnTopicExtractor
        if params["model_choice"] == "sk"
        else KtrainTopicExtractor
    )
    topic_extractor = topic_class(
        n_components=params["n_components"],
        n_features=params["n_features"],
        min_df=params["min_df"],
        max_df=params["max_df"],
    )

    # Fit topic extractor on available chunk texts
    texts_for_fitting = [item["chunk"] for item in all_chunks]
    topic_extractor.fit(texts_for_fitting)
    topics = topic_extractor.get_topics()

    print("Topics extracted:")
    print("\n".join(topics))

    # Clean topics using LLM with configurable model
    llm = LLM(model=params["llm_model"])
    cleaned_topics = llm.clean_topics(topics)

    print("Topics cleaned:")
    print("\n".join(cleaned_topics))

    # Ensure output folder exists
    os.makedirs(params["output_folder"], exist_ok=True)

    # Classify and save each text chunk with metadata including source_file
    for i, entry in enumerate(tqdm(all_chunks, desc="Generating Tags")):
        classification = llm.classify(entry["chunk"], cleaned_topics)
        output_data = {
            "chunk": entry["chunk"],
            "source_file": entry["source_file"],
            "classification": classification,
        }
        output_path = os.path.join(params["output_folder"], f"chunk_{i+1}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunk files to {params['output_folder']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline to load PDFs, split text, extract topics, and classify chunks."
    )
    parser.add_argument(
        "--pdf_folder", type=str, required=True, help="Path to folder containing PDFs"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512, help="Chunk size for text splitter"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=25, help="Chunk overlap for text splitter"
    )
    parser.add_argument(
        "--n_components", type=int, default=None, help="Number of topics to extract"
    )
    parser.add_argument(
        "--n_features", type=int, default=512, help="Max features for TopicExtractor"
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=2,
        help="Minimum document frequency for TopicExtractor",
    )
    parser.add_argument(
        "--max_df",
        type=float,
        default=0.95,
        help="Maximum document frequency for TopicExtractor",
    )
    parser.add_argument(
        "--llm_model", type=str, default="ollama:phi4", help="LLM model to use"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save chunk files with metadata",
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        choices=["ktrain", "sk"],
        default="ktrain",
        help='Choose topic extractor: "ktrain" for ktrain modeller or "sk" for sklearn model',
    )
    # ...existing argument definitions if any...

    args = parser.parse_args()
    main(vars(args))
