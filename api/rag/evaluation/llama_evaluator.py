import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from llama_index.core.llms import LLM
from llama_index.core import Response
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, MetadataMode
import nltk
from tqdm import tqdm

from rag.embeddings.embedding_factory import EmbeddingFactory
from rag.pipelines.base_pipeline import BasePipeline
from rag.utils.utils import get_unique_sources
import random

# Download NLTK resources if needed
nltk.download("punkt", quiet=True)


class RAGEvaluator:
    """A simplified evaluation framework for RAG pipelines."""

    def __init__(
        self,
        pipeline1: BasePipeline,
        pipeline2: BasePipeline,
        embed_model: str,
        llm: LLM,
    ):
        """
        Initialize the RAG evaluator.

        Args:
            pipeline1: First RAG pipeline to evaluate
            pipeline2: Second RAG pipeline to evaluate
            llm: LLM to use for evaluation
            similarity_threshold: Threshold for semantic similarity
            num_eval_questions: Number of evaluation questions to generate
        """
        self.pipeline1 = pipeline1
        self.pipeline2 = pipeline2
        self.llm = llm
        self.embed_model = EmbeddingFactory.create_embedding_model(embed_model)
        # Initialize evaluators
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm)
        self.relevancy_evaluator = RelevancyEvaluator(llm=self.llm)
        self.similarity_evaluator = SemanticSimilarityEvaluator(
            embed_model=self.embed_model
        )

        # Track evaluation data
        self.evaluation_questions = []
        self.results = {"pipeline1": pd.DataFrame(), "pipeline2": pd.DataFrame()}

    async def generate_evaluation_questions(
        self, nodes: List[BaseNode], persist_path: str = "./storage/dataset"
    ) -> List[Dict]:
        """Generate evaluation questions from the documents."""

        # Select a random sample of 100 nodes for evaluation
        if len(nodes) < 100:
            random_nodes = nodes
        else:
            random_nodes = random.sample(nodes, 100)

        dataset_generator = RagDatasetGenerator(
            nodes=random_nodes,
            llm=self.llm,
            num_questions_per_chunk=1,
            metadata_mode=MetadataMode.ALL,
            show_progress=True,
            question_gen_query=(
                "Generate exactly 1 concise question based solely on the provided context. "
                "The question should test specific knowledge from the document. "
                "Format your response as a single question only, with no introductions, explanations, or additional text. "
                "Do not include phrases like 'Based on the context' or 'According to the document'. "
            ),
        )

        print(
            "Number of chunks selected for dataset gen : ", len(dataset_generator.nodes)
        )

        file_path = os.path.join(
            persist_path, "-".join(get_unique_sources(random_nodes)) + ".json"
        )
        if os.path.exists(file_path):
            print(f"Loading evaluation questions from {file_path}")
            eval_dataset = LabelledRagDataset.from_json(file_path)
        else:
            eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            eval_dataset.save_json(file_path)
            print(f"Saved evaluation questions to {file_path}")

        self.evaluation_questions = [
            {"query": item.query, "relevant_docs": item.reference_contexts}
            for item in eval_dataset.examples
        ]
        return self.evaluation_questions

    async def calculate_similarity(
        self, relavant_context: str, retrieved_context: str
    ) -> bool:
        """Calculate semantic similarity between two texts"""
        similarity = await self.similarity_evaluator.aevaluate(
            query="", response=retrieved_context, reference=relavant_context
        )
        return similarity.passing

    async def evaluate_metrics(
        self,
        query: str,
        response: Response,
        retrieved_contexts: List[str],
        relevant_contexts: List[str],
    ):
        """Evaluate all metrics for a response."""
        metrics = {}

        # Contextual precision and recall
        relevant_retrieved = 0
        for context in retrieved_contexts:
            for relevant_context in relevant_contexts:
                is_similar = await self.calculate_similarity(relevant_context, context)
                if is_similar:
                    relevant_retrieved += 1
                    break

        # Calculate precision, recall, and F1
        precision = (
            relevant_retrieved / len(retrieved_contexts) if retrieved_contexts else 0
        )
        recall = (
            min(relevant_retrieved / len(relevant_contexts), 1.0)
            if relevant_contexts
            else 1.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        metrics["contextual_precision"] = precision
        metrics["contextual_recall"] = recall
        metrics["contextual_relevancy"] = f1

        # Answer relevancy
        relevancy_result = await self.relevancy_evaluator.aevaluate_response(
            query=query, response=response
        )
        metrics["answer_relevancy"] = float(relevancy_result.score)

        # Faithfulness
        faithfulness_result = await self.faithfulness_evaluator.aevaluate_response(
            query=query, response=response
        )
        metrics["faithfulness"] = float(faithfulness_result.score)

        # Hallucination
        if not relevant_contexts:
            metrics["hallucination"] = 0.5  # No ground truth to compare against
        else:
            hallucination_result = await self.faithfulness_evaluator.aevaluate(
                query=query, response=str(response), contexts=relevant_contexts
            )
            # Hallucination is the inverse of faithfulness to ground truth
            metrics["hallucination"] = 1.0 - float(hallucination_result.score)

        return metrics

    async def run_evaluation(self, num_questions: int = 20):
        """Run the evaluation on both pipelines."""
        # Generate evaluation questions if not already generated
        if not self.evaluation_questions:
            print("Generating evaluation questions...")
            await self.generate_evaluation_questions(self.pipeline1.get_nodes())

        results1 = {
            "query": [],
            "answer_relevancy": [],
            "contextual_precision": [],
            "contextual_recall": [],
            "contextual_relevancy": [],
            "faithfulness": [],
            "hallucination": [],
        }

        results2 = {
            "query": [],
            "answer_relevancy": [],
            "contextual_precision": [],
            "contextual_recall": [],
            "contextual_relevancy": [],
            "faithfulness": [],
            "hallucination": [],
        }

        # Select a random sample of 10 questions for evaluation
        if len(self.evaluation_questions) > num_questions:
            self.evaluation_questions = random.sample(
                self.evaluation_questions, num_questions
            )

        print(f"Evaluating {len(self.evaluation_questions)} questions...")
        for question_data in tqdm(self.evaluation_questions):
            query = question_data["query"]
            relevant_contexts = question_data["relevant_docs"]

            # Evaluate pipeline 1
            response1 = self.pipeline1.query(query)
            retrieved_nodes1 = getattr(response1, "source_nodes", []) or getattr(
                response1, "nodes", []
            )
            retrieved_contexts1 = [node.node.text for node in retrieved_nodes1]

            # Evaluate pipeline 2
            response2 = self.pipeline2.query(query)
            retrieved_nodes2 = getattr(response2, "source_nodes", []) or getattr(
                response2, "nodes", []
            )
            retrieved_contexts2 = [node.node.text for node in retrieved_nodes2]

            # Get metrics for both pipelines
            metrics1 = await self.evaluate_metrics(
                query, response1, retrieved_contexts1, relevant_contexts
            )
            metrics2 = await self.evaluate_metrics(
                query, response2, retrieved_contexts2, relevant_contexts
            )

            # Store results
            results1["query"].append(query)
            results2["query"].append(query)

            for metric in metrics1:
                results1[metric].append(metrics1[metric])
                results2[metric].append(metrics2[metric])

        # Convert to DataFrames
        self.results["pipeline1"] = pd.DataFrame(results1)
        self.results["pipeline2"] = pd.DataFrame(results2)

        return self.results

    def get_summary_statistics(self):
        """Get summary statistics of the evaluation results."""
        if self.results["pipeline1"].empty or self.results["pipeline2"].empty:
            return "Run evaluation first by calling run_evaluation()"

        # Calculate mean scores for each metric
        pipeline1_means = self.results["pipeline1"].mean(numeric_only=True)
        pipeline2_means = self.results["pipeline2"].mean(numeric_only=True)

        # Create summary DataFrame
        summary = pd.DataFrame(
            {
                "Pipeline 1": pipeline1_means,
                "Pipeline 2": pipeline2_means,
                "Difference": pipeline1_means - pipeline2_means,
            }
        )

        # Add winning pipeline column (for hallucination, lower is better)
        winners = []
        for idx, diff in enumerate(summary["Difference"]):
            metric = summary.index[idx]
            if metric == "hallucination":
                winners.append(
                    "Pipeline 1" if diff < 0 else "Pipeline 2" if diff > 0 else "Tie"
                )
            else:
                winners.append(
                    "Pipeline 1" if diff > 0 else "Pipeline 2" if diff < 0 else "Tie"
                )

        summary["Winner"] = winners
        return summary

    def visualize_results(
        self, persist_path: str = "./storage/results", output_file=None
    ):
        """Visualize the evaluation results as radar and bar charts."""
        summary = self.get_summary_statistics()
        if isinstance(summary, str):
            return summary

        # Get metrics (excluding query)
        metrics = [
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "faithfulness",
        ]

        # For hallucination, lower is better, so we invert it
        pipeline1_hallucination = 1 - self.results["pipeline1"]["hallucination"].mean()
        pipeline2_hallucination = 1 - self.results["pipeline2"]["hallucination"].mean()

        # Get scores
        pipeline1_scores = [
            self.results["pipeline1"][metric].mean() for metric in metrics
        ]
        pipeline1_scores.append(pipeline1_hallucination)

        pipeline2_scores = [
            self.results["pipeline2"][metric].mean() for metric in metrics
        ]
        pipeline2_scores.append(pipeline2_hallucination)

        # Add hallucination (inverted) to metrics
        metrics.append("non_hallucination")

        # Set up the radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        pipeline1_scores += pipeline1_scores[:1]  # Close the loop
        pipeline2_scores += pipeline2_scores[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        # Plot the scores
        ax.plot(angles, pipeline1_scores, "o-", linewidth=2, label="Pipeline 1")
        ax.plot(angles, pipeline2_scores, "o-", linewidth=2, label="Pipeline 2")
        ax.fill(angles, pipeline1_scores, alpha=0.1)
        ax.fill(angles, pipeline2_scores, alpha=0.1)

        # Set the labels
        labels = metrics + [metrics[0]]  # Close the loop
        ax.set_xticks(angles)
        ax.set_xticklabels([label.replace("_", " ").title() for label in labels])

        # Set the limit of each dimension
        ax.set_ylim(0, 1)

        # Add legend
        plt.legend(loc="upper right")

        # Add title
        plt.title("RAG Pipeline Evaluation Comparison", size=15)

        # Save if output file is specified
        if output_file:
            plt.savefig(
                os.path.join(persist_path, output_file), dpi=300, bbox_inches="tight"
            )

        # Show the plot
        plt.tight_layout()
        plt.show()

        return summary

    async def evaluate(
        self,
        num_questions: int = 20,
        output_file="rag_evaluation_results.pdf",
    ) -> pd.DataFrame:
        """
        Run a comprehensive evaluation of the two configured RAG pipelines and generate visualizations.

        This function executes the full evaluation workflow by running all test queries against both
        pipelines, computing evaluation metrics, generating a summary comparison, and visualizing the results.

        Args:
            output_file: Path to save visualization results (default: "rag_evaluation_results.pdf")

        Returns:
            pd.DataFrame: Summary statistics comparing the performance of both pipelines
        """

        print("Running evaluation...")
        await self.run_evaluation(num_questions)

        print("\nSummary Statistics:")
        summary = self.get_summary_statistics()
        print(summary)

        print("\nGenerating visualizations...")
        self.visualize_results(output_file=output_file)

        return summary
