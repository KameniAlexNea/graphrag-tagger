# **graphrag-tagger**

*A lightweight toolkit for extracting topics from PDFs and visualizing their connections using graphs.*

## **Overview**

`graphrag-tagger` automates topic extraction from PDF documents and builds graphs to visualize relationships between text segments. It offers a modular pipeline for processing text, applying topic modeling, refining results with an LLM, and constructing a graph-based representation of topic similarities.

### **Key Features**

✅ **PDF Processing** – Extracts text from PDFs efficiently.✅ **Text Segmentation** – Splits extracted text into manageable chunks.✅ **Topic Modeling** – Supports two methods:

- **Scikit-learn**: Classic **Latent Dirichlet Allocation (LDA)** for topic extraction.
- **ktrain**: A deep-learning-based approach with vocabulary filtering.
  ✅ **LLM-Powered Refinement** – Uses a language model to clean and classify topics.
  ✅ **Graph Construction** – Builds topic similarity graphs using **network analysis**.

### **Core Dependencies**

- **PyMuPDF** – Extracts text from PDF files.
- **scikit-learn & ktrain** – Performs topic modeling.
- **LLM Client** – Enhances and refines extracted topics.
- **networkx** – Constructs and analyzes graphs.

---

## **Installation**

Ensure you have Python installed, then install the package locally:

```bash
pip install graphrag-tagger
```

---

## **Usage**

### **Extract Topics from PDFs**

Run the topic extraction pipeline on a folder of PDFs:

```bash
python -m graphrag_tagger.tagger \
    --pdf_folder /path/to/pdfs \
    --output_folder /path/to/output \
    --chunk_size 512 \
    --chunk_overlap 25 \
    --n_features 512 \
    --min_df 2 \
    --max_df 0.95 \
    --llm_model ollama:phi4 \
    --model_choice sk
```

### **Build a Topic Similarity Graph**

Generate a graph from the extracted topics:

```bash
python -m graphrag_tagger.build_graph \
    --input_folder /path/to/output \
    --output_folder /path/to/graph \
    --threshold_percentile 97.5
```

---

## **How It Works**

1️⃣ **PDF Processing** – Extracts raw text from documents.2️⃣ **Text Segmentation** – Divides the text into structured chunks.3️⃣ **Topic Modeling** – Uses either LDA or ktrain-based modeling to extract key topics.4️⃣ **LLM-Based Refinement** – Cleans and classifies topics for better accuracy.5️⃣ **Graph Construction** – Builds a network where:

- **Nodes** represent text chunks.
- **Edges** represent topic similarities.
- The graph reveals **clusters** and **connections** between document sections.

---

## **Contributing**

Contributions are welcome! Feel free to submit issues or pull requests.
