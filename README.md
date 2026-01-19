# Med-CoReasoner

# Med-CoReasoner

**Med-CoReasoner** is a multilingual medical reasoning agent that combines cross-lingual concept fusion with retrieval-augmented generation (RAG) for clinical question answering. It processes medical questions in both English and the target language, extracts key medical concepts, and fuses them using embedding-based similarity to produce accurate, knowledge-grounded answers.

## Architecture

```
Medical Question (Multiple Languages)
        │
        ▼
    ┌─────────────────────────────────────────┐
    │              REASONER                    │
    │   English Reasoning ◄─► Local Reasoning │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │              EXTRACTOR                   │
    │   English Concepts ◄─► Local Concepts   │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │              RETRIEVER                   │
    │   RAG: Medical Documents (MSD Corpus)   │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │               FUSION                     │
    │   Embedding-based Concept Merging       │
    │   (BGE-M3 Similarity Matching)          │
    └─────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────┐
    │              GENERATOR                   │
    │   Final Answer + Reasoning Trace        │
    └─────────────────────────────────────────┘
```

## Features

- **Multilingual Support**: English, Chinese, Japanese, Korean, German, French, Spanish, Italian, Thai, Swahili, Yoruba, Zulu
- **Cross-lingual Concept Fusion**: Combines reasoning from English and target language using BGE-M3 embeddings
- **RAG Integration**: Retrieves and reranks medical documents using BGE-M3 and BGE-Reranker
- **Multiple Task Types**: Medical QA, BioNLI (Natural Language Inference), LiveQA (daily-life medical questions)
- **Concurrent Processing**: Multi-threaded execution for efficient batch processing

## Requirements

### Dependencies

```bash
pip install openai anthropic transformers torch numpy faiss-cpu
pip install llama-index llama-index-embeddings-huggingface
pip install FlagEmbedding datasets pandas tqdm python-dotenv
```

### Environment Variables

Create a `.env` file in the `codes/` directory:

```bash
OPENAI_API_KEY="your-openai-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional: custom endpoint
ANTHROPIC_API_KEY="your-anthropic-api-key"
DEEPSEEK_API_KEY="your-deepseek-api-key"
GEMINI_API_KEY="your-gemini-api-key"
```

### Medical Corpus Setup

The RAG system requires a medical corpus. Place your corpus in:

```
codes/CLARA/corpus/
└── msd/
    ├── en/
    │   ├── chunk/
    │   │   └── *.jsonl  # Medical documents
    │   └── index/
    │       └── BAAI/bge-m3/  # Vector index (auto-generated)
    └── {language}/
        ├── chunk/
        └── index/
```

Each JSONL file should contain documents with `title` and `content` fields.

## Usage

### Running Medical QA Benchmarks

```bash
cd codes
python main.py
```

**Configuration** (edit `main.py`):

```python
# Model selection
model_name = "gpt-4o"  # Options: gpt-4o, gpt-5.1, claude-3-5-sonnet, deepseek-chat, etc.

# Datasets
dataset_names = [
    "global-mmlu",
    "mmlu-prox",
]

# Target languages
language_settings = [
    "en",   # English
    "zh",   # Chinese
    "ja",   # Japanese
    "ko",   # Korean
    "de",   # German
    "fr",   # French
    "es",   # Spanish
    "it",   # Italian
]

# Processing parameters
top_k = 5           # Number of retrieved documents
max_workers = 10    # Parallel threads
```



Constructs multilingual training datasets for fine-tuning.

## Output Format

Results are saved as JSONL files in:

```
benchmark/{dataset_name}/results/run_round_{N}/{language}/local_{model}_results.jsonl
```

Each line contains:

```json
{
  "question": "Medical question text",
  "options": "A. Option1\nB. Option2\nC. Option3\nD. Option4",
  "answer_idx": "A",
  "en_answer": "A",
  "en_reasoning": "Detailed English reasoning process...",
  "local_answer": "A",
  "local_reasoning": "Reasoning in target language...",
  "en_concept_chain": ["concept1", "concept2", "concept3"],
  "local_concept_chain": ["概念1", "概念2", "概念3"],
  "fused_concept_chain": ["merged_concept1", "merged_concept2", ...],
  "rejected_concepts": ["concept_not_similar_enough"],
  "response": "Final answer with verified reasoning"
}
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-5, gpt-5.1, o3, o3-mini |
| Anthropic | claude-3-5-sonnet, claude-3-5-haiku |
| DeepSeek | deepseek-chat |
| Google | gemini-2.5-pro, gemini-2.5-flash |
| Local | Qwen, LLaMA, DeepSeek (via transformers) |

## Supported Datasets

- **global-mmlu**: Global MMLU medical questions
- **mmlu-prox**: MMLU proxy dataset
- **liveqa**: MultiMed-X LiveQA dataset
- **bionli**: MultiMed-X NLI dataset

## Project Structure

```
codes/
├── main.py                 # Main entry point for medical QA
├── load_model.py          # Model loading utilities
├── load_datasets.py       # Dataset loading functions
├── utils.py               # Helper functions
├── evaluate.py            # Evaluation metrics
├── rag.py             # RAG system implementation
```

## Key Parameters

### Med-CoReasoner Agent

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `gpt-4o` | LLM model to use |
| `temperature` | `0.7` | Sampling temperature |
| `corpus_name` | `msd` | Medical corpus name |
| `retriever_name` | `BAAI/bge-m3` | Embedding model for retrieval |
| `reranker_name` | `BAAI/bge-reranker-base` | Reranker model |
| `if_rerank` | `True` | Enable reranking |
| `language` | `en` | Target language code |
| `db_dir` | `./corpus` | Corpus directory path |

<!-- ### Concept Fusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.5` | Similarity threshold for concept fusion |
 -->


## License
MultiMed-X comprises two tasks spanning seven non-English languages, and all data is distributed under the CC BY 4.0 license.

The Usage of retrived corpus from MSD are under official permission.