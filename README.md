# graphrag-tagger

graphrag-tagger is a lightweight toolkit designed for topic extraction and tagging in GraphRAG applications. The project leverages classic LDA techniques along with modern LLM capabilities to generate, refine, and classify topic labels.

## Installation

Install the package via pip:
```bash
pip install .
```

## Components

### Topic Modeling
- **Sklearn Implementation**  
  Located in `graphrag_tagger/lda/sk_modelling.py`, this module:
  - Uses scikit-learn's Latent Dirichlet Allocation (LDA) to extract topics.
  - Auto-determines the number of topics if not specified.
  - Provides methods to fit on text data, transform new text, and extract themes based on weight thresholds.

- **KTrain Implementation**  
  Located in `graphrag_tagger/lda/kt_modelling.py`, this module:
  - Uses ktrain's topic modeling functions.
  - Builds an LDA model with configurable vocabulary and frequency-based filtering.
  - Offers functions for topic extraction, filtering texts, and transforming documents into topic distributions.

### LLM Integration
- **Prompt Templates**  
  Found in `graphrag_tagger/chat/prompts.py`, these define the instructions for:
  - Refining and cleaning up messy topics.
  - Classifying text according to candidate topics.
  
- **JSON Parser**  
  Implemented in `graphrag_tagger/chat/parser.py`, this module:
  - Provides robust JSON parsing by handling common formatting issues.
  - Extracts relevant JSON blocks from verbose text.

- **LLM Client**  
  Located in `graphrag_tagger/chat/llm.py`, this module:
  - Integrates with an LLM (via aisuite) to clean topics and classify document excerpts.
  - Uses the prompt templates along with the JSON parser to deliver refined outputs.

## Usage Example
