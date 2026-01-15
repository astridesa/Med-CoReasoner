import os
import torch
import tqdm
import json
import numpy as np

import torch.nn as nn

_original_to = nn.Module.to


def _patched_to(self, *args, **kwargs):
    """Patched version of to() that uses to_empty() for meta tensors"""
    try:
        return _original_to(self, *args, **kwargs)
    except NotImplementedError as e:
        if "meta tensor" in str(e):
            # Use to_empty() for meta tensors
            device = args[0] if args else kwargs.get("device", None)
            if device is not None:
                return self.to_empty(device=device)
        raise


nn.Module.to = _patched_to

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# from llama_index.postprocessor.cross_encoder_rerank import CrossEncoderRerank
from llama_index.core.postprocessor import SentenceTransformerRerank


class RAG:
    def __init__(
        self,
        corpus_name="msd",
        retriever_name="BAAI/bge-m3",
        reranker_name="BAAI/bge-reranker-base",
        language="en",
        db_dir="./corpus",
        auto_load_index=True,
    ):
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.reranker_name = reranker_name
        self.language = language
        self.db_dir = db_dir
        self.index_path = os.path.join(
            self.db_dir,
            f"{self.corpus_name}/{self.language}/index/{self.retriever_name}",
        )
        self.chunk_path = os.path.join(
            self.db_dir, f"{self.corpus_name}/{self.language}/chunk"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load embedding model
        model_kwargs = {
            "torch_dtype": (
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            "trust_remote_code": True,
        }

        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.retriever_name,
            device=self.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )

        # Load reranker once during initialization to avoid repeated loading
        # Pass device parameter to avoid meta tensor issues
        self.reranker = SentenceTransformerRerank(
            model=self.reranker_name,
            device=self.device,
        )

        # Load or build index
        self.index = None
        if auto_load_index:
            if os.path.exists(self.index_path):
                print(f"Loading existing index: {self.index_path}")
                self._load_index()
            else:
                print(f"Index does not exist and will be built on first use: {self.index_path}")

    # load articles
    def _load_articles(self, title_key="title", content_key="content"):
        """Load articles from all jsonl files in chunk directory"""
        articles = []

        # Check if chunk directory exists
        if not os.path.exists(self.chunk_path):
            raise FileNotFoundError(f"Chunk directory not found: {self.chunk_path}")

        # Get all jsonl files
        jsonl_files = [f for f in os.listdir(self.chunk_path) if f.endswith(".jsonl")]

        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {self.chunk_path}")

        print(f"Found {len(jsonl_files)} JSONL files")

        # Iterate over all jsonl files and load
        for filename in jsonl_files:
            articles_file = os.path.join(self.chunk_path, filename)

            with open(articles_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): 
                        continue

                    article = json.loads(line.strip())
                    text = article.get(content_key, "")
                    title = article.get(title_key, "")

                    # Skip empty documents
                    if not text and not title:
                        continue

                    # Extract metadata
                    meta_data = {
                        k: v
                        for k, v in article.items()
                        if k not in [title_key, content_key]
                    }
                    doc = Document(text=title + "\n" + text, metadata=meta_data)
                    articles.append(doc)

        return articles

    def _build_index(self):
        """Build vector index from articles using SimpleVectorStore"""
        articles = self._load_articles()

        print("Generating embeddings for documents...")
        index = VectorStoreIndex.from_documents(
            articles,
            embed_model=self.embedding_model,
            show_progress=True,
        )

        index.storage_context.persist(self.index_path)
        print(f"✓ Index has been built and saved to: {self.index_path}")

        self.index = index
        return index

    def _load_index(self):
        """Load existing index from storage - standard method"""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
            self.index = load_index_from_storage(
                storage_context=storage_context, embed_model=self.embedding_model
            )
            print(f"✓ Index loaded successfully")
        except Exception as e:
            print(f"✗ Index loading failed: {e}")
            print(f"   Index path: {self.index_path}")
            import traceback

            traceback.print_exc()
            self.index = None

    def retrieve(self, query, top_k=50, rerank_k=20, if_rerank=True):
        """Retrieve top-k relevant articles for a given query"""

        # Create retriever
        retriever = self.index.as_retriever(similarity_top_k=top_k)

        # Retrieve documents
        nodes = retriever.retrieve(query)

        # Optional reranking
        if if_rerank:
            self.reranker.top_n = rerank_k
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query)
            results = [
                {
                    "content": node.node.text,
                    "metadata": node.node.metadata,
                    "score": node.score,
                }
                for node in reranked_nodes
            ]
        else:
            results = [
                {"content": node.text, "metadata": node.metadata, "score": node.score}
                for node in nodes[:rerank_k]  # Limit the number of returned results
            ]

        return results


if __name__ == "__main__":
    corpus_name = "msd"
    retriever_name = "BAAI/bge-m3"
    reranker_name = "BAAI/bge-reranker-base"
    language = "en"
    db_dir = "./corpus"

    rag = RAG(
        corpus_name=corpus_name,
        retriever_name=retriever_name,
        reranker_name=reranker_name,
        language=language,
        db_dir=db_dir,
        auto_load_index=True,
    )

    print("\n" + "=" * 60)
    print("step 1 : build index")
    print("=" * 60)
    rag._build_index()

    print("\n" + "=" * 60)
    print("step 2: test retrieval")
    print("=" * 60)
    query = "What are the symptoms of diabetes?"
    results = rag.retrieve(query, top_k=10, rerank_k=5, if_rerank=True)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} relevant documents:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result['score']:.4f}]")
        print(f"   Content preview: {result['content'][:100]}...")
        print(f"   Metadata: {result['metadata']}")
        print()
