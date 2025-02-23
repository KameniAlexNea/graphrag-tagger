# graphrag-tagger

graphrag-tagger is a lightweight toolkit for extracting topics from PDF documents and then building graphs to visualize connections between text segments.

## Overview

This package processes PDF files by:

- Extracting text using a PDF library.
- Splitting the text into manageable chunks.
- Performing topic modeling with two alternative implementations:
  - A scikit-learn based approach using classic LDA.
  - A ktrain-powered approach with configurable vocabulary filtering.
- Refining topics using an LLM to clean and classify the extracted topics.
- Building graphs to show relationships between text chunks based on topic similarity, leveraging network analysis.

Key libraries used include:

- PyMuPDF for PDF processing.
- scikit-learn and ktrain for topic modeling.
- An LLM client for natural language processing.
- networkx for graph construction and analysis.

## Installation

Build the package using the recommended tool:

```bash
python -m build
```

Then install locally:

```bash
pip install .
```

## Usage

### Command-Line Interface

You can run the pipeline to process PDFs and extract topic information:

```bash
python -m graphrag_tagger.tagger --pdf_folder /path/to/pdfs --output_folder /path/to/output --chunk_size 512 --chunk_overlap 25 --n_features 512 --min_df 2 --max_df 0.95 --llm_model ollama:phi4 --model_choice sk
```

And build a graph from the output:

```bash
python -m graphrag_tagger.build_graph --input_folder /path/to/output --output_folder /path/to/graph --threshold_percentile 97.5
```

## How It Works

1. PDF files are read and their text is extracted.
2. Text is segmented into chunks based on specified sizes.
3. Topic modeling algorithms analyze these chunks to generate candidate topics.
4. A language model cleans and refines these topics.
5. A graph is constructed where nodes represent text chunks and edges represent shared topic contributions, allowing you to visualize clusters and connections.
