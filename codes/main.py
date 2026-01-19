import os
from pathlib import Path
import sys
import concurrent.futures
import numpy as np

os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from FlagEmbedding import BGEM3FlagModel

try:
    import transformers

    transformers.modeling_utils.LOW_CPU_MEM_USAGE_DEFAULT = False
except Exception:
    pass


sys.path.append(str(Path(__file__).resolve().parent.parent))
from load_datasets import (
    load_medagents,
    load_mmedbench,
    load_medcasereasoning,
    load_curebench,
    load_global_mmlu,
    load_mmlu_prox,
)
from utils import parse_json, write_jsonl_with_lock
from agent_think.rag import RAG
from load_model import ModelLoader
from tqdm import tqdm
from evaluate import evaluate_multi_options
from med_prompts import *


def insert_concepts(query_list, candidates, model, threshold=0.5):
    # Compute embeddings
    all_texts = candidates + query_list
    all_embeddings = model.encode(all_texts, batch_size=64, return_dense=True)[
        "dense_vecs"
    ]
    all_embeddings = all_embeddings / np.linalg.norm(
        all_embeddings, axis=1, keepdims=True
    )

    candidate_embeddings = all_embeddings[: len(candidates)]
    query_embeddings = all_embeddings[len(candidates) :]

    inserted_queries = []
    rejected_queries = []
    current_candidates = candidates.copy()
    current_embeddings = candidate_embeddings.copy()

    for query_idx, (query, query_emb) in enumerate(zip(query_list, query_embeddings)):
        # calculate similarities
        similarities = current_embeddings @ query_emb

        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]

        if max_similarity >= threshold:
            # determine insert position
            if max_sim_idx == 0:
                insert_pos = 1
            elif max_sim_idx == len(current_candidates) - 1:
                insert_pos = max_sim_idx
            else:
                left_avg = np.mean(similarities[:max_sim_idx])
                right_avg = np.mean(similarities[max_sim_idx + 1 :])
                insert_pos = max_sim_idx if left_avg > right_avg else max_sim_idx + 1

            # insert text and embedding
            current_candidates.insert(insert_pos, query)
            current_embeddings = np.insert(
                current_embeddings, insert_pos, query_emb, axis=0
            )

            inserted_queries.append(
                {
                    "query": query,
                    "max_similarity": float(max_similarity),
                    "insert_position": insert_pos,
                }
            )
        else:
            rejected_queries.append(
                {
                    "query": query,
                    "max_similarity": float(max_similarity),
                    "most_similar": current_candidates[max_sim_idx],
                }
            )

    return current_candidates, inserted_queries, rejected_queries


class CoReasoner:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        reasoning_effort: str = "low",
        seed: int = 42,
        corpus_name: str = "msd",
        retriever_name: str = "BAAI/bge-m3",
        reranker_name: str = "BAAI/bge-reranker-base",
        if_rerank: bool = True,
        language: str = "en",
        db_dir: str = "./corpus",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.model_loader = ModelLoader(model_name)
        self.corpus_name = corpus_name
        self.retriever_name = retriever_name
        self.reranker_name = reranker_name
        self.if_rerank = if_rerank
        self.db_dir = db_dir
        self.rag_system = RAG(
            corpus_name=corpus_name,
            retriever_name=retriever_name,
            reranker_name=reranker_name,
            language=language,
            db_dir=db_dir,
            auto_load_index=True,
        )
        self.en_rag_system = RAG(
            corpus_name=corpus_name,
            retriever_name=retriever_name,
            reranker_name=reranker_name,
            language="en",
            db_dir=db_dir,
            auto_load_index=True,
        )
        self.similarity_model = BGEM3FlagModel(retriever_name, use_fp16=True)
        if language == "en":
            self.language = "ENGLISH"
        elif language == "zh":
            self.language = "CHINESE"
        elif language == "ko":
            self.language = "KOREAN"
        elif language == "ja":
            self.language = "JAPANESE"
        elif language == "de":
            self.language = "GERMAN"
        elif language == "es":
            self.language = "SPANISH"
        elif language == "fr":
            self.language = "FRENCH"
        elif language == "it":
            self.language = "ITALIAN"

    def _load_data(self, dataset_name: str, subset: str, split: str = "test"):
        if dataset_name == "medagents":
            return load_medagents(subset, split)
        elif dataset_name == "mmedbench":
            return load_mmedbench(subset, split)
        elif dataset_name == "medcasereasoning":
            return load_medcasereasoning(subset, split)
        elif dataset_name == "cure-bench":
            return load_curebench(subset, split)
        elif dataset_name == "global-mmlu":
            return load_global_mmlu(subset, split, is_lite=False)
        elif dataset_name == "mmlu-prox":
            return load_mmlu_prox(subset, split, is_lite=False)
        elif dataset_name == "global-mmlu-lite":
            return load_global_mmlu(subset, split, is_lite=True)
        elif dataset_name == "mmlu-prox-lite":
            return load_mmlu_prox(subset, split, is_lite=True)
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

    def _reasoner(self, data, prompt: str, language: str = "en"):
        messages = [
            {
                "role": "system",
                "content": "You are a cautious medical expert AI assistant.",
            },
            {
                "role": "user",
                "content": prompt.format(
                    question=data["question"],
                    options=data.get("options", ""),
                    language=language,
                ),
            },
        ]
        response = self.model_loader.get_response(
            messages, temperature=self.temperature
        )
        response = parse_json(response)
        answer = response.get("answer", "")
        reasoning = response.get("reasoning", "")
        return answer, reasoning

    def _extractor(self, reasoning, prompt: str, language: str = "en"):
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert in logical information extraction.",
            },
            {
                "role": "user",
                "content": prompt.format(reasoning_trace=reasoning, language=language),
            },
        ]
        response = self.model_loader.get_response(
            messages, temperature=self.temperature
        )
        reasoning_chain = parse_json(response)
        if not isinstance(reasoning_chain, list):
            reasoning_chain = []
        return reasoning_chain

    def _retriever(self, query: str, top_k: int = 5, language: str = "en"):
        if language == "en":
            results = self.en_rag_system.retrieve(
                query, top_k=10, rerank_k=top_k, if_rerank=self.if_rerank
            )
        else:
            results = self.rag_system.retrieve(
                query, top_k=10, rerank_k=top_k, if_rerank=self.if_rerank
            )
        retrieved_docs = [result["content"] for result in results]
        return retrieved_docs

    def _translator(
        self,
        data,
        documents: list,
        prompt: str,
        source_language: str = "en",
        target_language: str = "en",
    ):
        question = data["question"]
        options = data.get("options", "")
        documents = "\n\n".join(
            ["Document {}:\n{}".format(i + 1, doc) for i, doc in enumerate(documents)]
        )
        messages = [
            {
                "role": "system",
                "content": "You are a professional medical translator AI assistant.",
            },
            {
                "role": "user",
                "content": prompt.format(
                    question=question,
                    options=options,
                    content=documents,
                    source_language=source_language,
                    target_language=target_language,
                ),
            },
        ]
        response = self.model_loader.get_response(
            messages, temperature=self.temperature
        )
        return response

    def _generator(
        self,
        data,
        reasoning_chain: str,
        context: str,
        prompt: str,
        language: str = "en",
    ):
        messages = [
            {
                "role": "system",
                "content": "You are a careful and conservative medical expert AI assistant.",
            },
            {
                "role": "user",
                "content": prompt.format(
                    question=data["question"],
                    options=data.get("options", ""),
                    concept_chain=reasoning_chain,
                    context=context,
                    language=language,
                ),
            },
        ]
        response = self.model_loader.get_response(
            messages, temperature=self.temperature
        )
        return response

    def process(
        self,
        data,
        en_data,
        reasoning_prompt: str,
        extraction_prompt: str,
        generation_prompt: str,
        translation_prompt: str,
        top_k: int = 5,
        output_file: str = None,
    ):
        question = data["question"]
        options = data.get("options", "")
        en_question = en_data["question"]
        en_options = en_data.get("options", "")
        # Step 1: Reasoning
        en_anwer, en_reasoning = self._reasoner(
            en_data, reasoning_prompt, language="ENGLISH"
        )
        local_answer, local_reasoning = self._reasoner(
            data, reasoning_prompt, language=self.language
        )
        # Step 2: Extract key concepts from reasoning trace
        en_concept_chain = self._extractor(
            en_reasoning, extraction_prompt, language="ENGLISH"
        )
        local_concept_chain = self._extractor(
            local_reasoning, extraction_prompt, language=self.language
        )
        # Step 3: Retrieve relevant documents based on key concepts
        en_retrieved_docs = []
        local_retrieved_docs = []
        # for concept in en_concept_chain:
        #     docs = self._retriever(concept, top_k=top_k, language="en")
        #     en_retrieved_docs.extend(docs)
        # for concept in local_concept_chain:
        #     docs = self._retriever(concept, top_k=top_k, language=self.language)
        #     local_retrieved_docs.extend(docs)
        en_retrieved_docs = self._retriever(en_question, top_k=top_k, language="en")
        local_retrieved_docs = self._retriever(
            question, top_k=top_k, language=self.language
        )
        retrieved_docs = list(set(en_retrieved_docs + local_retrieved_docs))
        referenced_context = "\n\n".join(retrieved_docs)

        # Step 4: Translate the retrieved documents if necessary (optional)
        # tranlated_retrieved_docs = self._translator(
        #     data,
        #     local_retrieved_docs,
        #     translation_prompt,
        #     source_language=self.language,
        #     target_language="ENGLISH",
        # )

        # referenced_context = "\n\n".join(en_retrieved_docs + tranlated_retrieved_docs)

        # Step 5: fuse the reasoning chain
        if len(local_concept_chain) == 0 or len(en_concept_chain) == 0:
            fused_concept_chain = []
            inserted_concepts = []
            rejected_concepts = []
        else:
            fused_concept_chain, inserted_concepts, rejected_concepts = insert_concepts(
                local_concept_chain,
                en_concept_chain,
                self.similarity_model,
                threshold=0.7,
            )

            rejected_concepts = [v["query"] for v in rejected_concepts]

        # step 6: Generate final answer based on fused concept chain and referenced documents
        response = self._generator(
            data,
            fused_concept_chain,
            referenced_context,
            generation_prompt,
            language=self.language,
        )
        final_results = {
            "question": data["question"],
            "options": data.get("options", ""),
            "answer_idx": data.get("answer_idx", 0),
            "en_reasoning": en_reasoning,
            "local_reasoning": local_reasoning,
            "en_concept_chain": en_concept_chain,
            "local_concept_chain": local_concept_chain,
            "fused_concept_chain": fused_concept_chain,
            "response": response,
        }
        if output_file is not None:
            write_jsonl_with_lock(output_file, final_results)
        return final_results

    def run_clara(
        self,
        dataset_name: str,
        subset: str,
        split: str = "test",
        reasoning_prompt: str = REASONING_PROMPT,
        extraction_prompt: str = EXTRACTION_PROMPT,
        generation_prompt: str = GENERATION_PROMPT,
        translation_prompt: str = TRANSLATION_PROMPT,
        top_k: int = 5,
        max_workers: int = 4,
        output_file: str = None,
    ):
        dataset = self._load_data(dataset_name, subset, split)
        en_dataset = self._load_data(dataset_name, "en", split)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item, en_item in zip(dataset, en_dataset):
                futures.append(
                    executor.submit(
                        self.process,
                        item,
                        en_item,
                        reasoning_prompt,
                        extraction_prompt,
                        generation_prompt,
                        translation_prompt,
                        top_k,
                        output_file,
                    )
                )
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing data",
            ):
                result = future.result()
                results.append(result)
        return results


if __name__ == "__main__":
    model_name = "gpt-5.1"

    dataset_names = [
        "global-mmlu",
        # "mmlu-prox"
    ]
    language_settings = [
        # "en",
        # "zh",
        # "ko",
        "ja",
        "de",
        "fr",
        "es",
        "it",
    ]
    for dataset_name in dataset_names:
        for language in language_settings:
            coreasoner_agent = CoReasoner(
                model_name=model_name,
                temperature=0.7,
                reasoning_effort="low",
                seed=420,
                corpus_name="msd",
                retriever_name="BAAI/bge-m3",
                reranker_name="BAAI/bge-reranker-base",
                if_rerank=True,
                language=language,
                db_dir="./corpus",
            )
            subset = language
            split = "test"
            run_round = "1"
            reasoning_prompt = REASONING_PROMPT
            extraction_prompt = EXTRACTION_PROMPT
            generation_prompt = GENERATION_PROMPT
            translation_prompt = TRANSLATION_PROMPT
            top_k = 5
            max_workers = 10
            output_file = os.path.join(
                f"benchmark/{dataset_name}/results",
                f"run_round_{run_round}",
                f"{language}",
                f"{model_name}_results.jsonl",
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                pass
            coreasoner_agent.run_clara(
                dataset_name=dataset_name,
                subset=subset,
                split=split,
                reasoning_prompt=reasoning_prompt,
                extraction_prompt=extraction_prompt,
                generation_prompt=generation_prompt,
                translation_prompt=translation_prompt,
                top_k=top_k,
                max_workers=max_workers,
                output_file=output_file,
            )
            accuracy = evaluate_multi_options(output_file)
            print(
                f"Dataset: {dataset_name}, Language: {language}, Model: {model_name}, Accuracy: {accuracy}"
            )
