import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import random

from ragas.testset import TestsetGenerator, Testset
from ragas.integrations.llama_index import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LlamaIndexLLMWrapper

from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, Document

from rag.embeddings.embedding_factory import EmbeddingFactory
from rag.pipelines.base_pipeline import BasePipeline
from rag.utils.utils import get_unique_sources


class RAGASEvaluator:
    """An evaluation framework for RAG pipelines using RAGAS metrics."""

    def __init__(
        self,
        pipeline1: BasePipeline,
        pipeline2: BasePipeline,
        embed_model: str,
        llm: LLM,
    ):
        """
        Initialize the RAGAS evaluator.

        Args:
            pipeline1: First RAG pipeline to evaluate
            pipeline2: Second RAG pipeline to evaluate
            embed_model: Name of the embedding model to use
            llm: LLM to use for evaluation
        """
        self.pipeline1 = pipeline1
        self.pipeline2 = pipeline2
        self.llm = llm
        self.embed_model = EmbeddingFactory.create_embedding_model(embed_model)

        # Initialize evaluator LLM wrapper
        self.evaluator_llm = LlamaIndexLLMWrapper(self.llm)

        # Initialize metrics
        self.metrics = [
            Faithfulness(llm=self.evaluator_llm),
            AnswerRelevancy(llm=self.evaluator_llm),
            ContextPrecision(llm=self.evaluator_llm),
            ContextRecall(llm=self.evaluator_llm),
        ]

        # Create test generator
        self.test_generator = TestsetGenerator.from_llama_index(
            llm=self.llm,
            embedding_model=self.embed_model,
        )

        # Track evaluation data
        self.evaluation_questions = []
        self.results = {"pipeline1": pd.DataFrame(), "pipeline2": pd.DataFrame()}

    async def generate_evaluation_questions(
        self,
        nodes: List[BaseNode],
        testset_size: int = 20,
        persist_path: str = "./storage",
    ):
        """Generate evaluation questions from documents using RAGAS."""

        # Select a random sample of nodes for evaluation
        if len(nodes) > 100:
            random_nodes = random.sample(nodes, 100)
        else:
            random_nodes = nodes

        # Convert nodes to Documents for RAGAS
        documents = [
            Document(text=node.text, metadata=node.metadata) for node in random_nodes
        ]

        file_path = os.path.join(
            persist_path,
            "ragas_dataset",
            "-".join(get_unique_sources(random_nodes)) + ".json",
        )

        print(f"Generating testset with {testset_size} questions")

        # Check if saved dataset exists
        if os.path.exists(file_path):
            print(f"Loading evaluation questions from {file_path}")
            testset_df = pd.read_json(file_path)
            testset = Testset.from_pandas(testset_df)
        else:
            # Generate testset with RAGAS
            testset = self.test_generator.generate_with_llamaindex_docs(
                documents,
                testset_size=testset_size,
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            testset.to_pandas().to_json(file_path)
            print(f"Saved evaluation questions to {file_path}")

        self.evaluation_questions = testset.to_evaluation_dataset()
        return self.evaluation_questions

    async def run_evaluation(
        self, num_questions: int = 20, persist_path: str = "./storage/results"
    ):
        """Run the evaluation on both pipelines using RAGAS metrics."""
        # Generate evaluation questions if not already generated
        if not self.evaluation_questions:
            print("Generating evaluation questions...")
            await self.generate_evaluation_questions(
                self.pipeline1.get_nodes(), testset_size=num_questions
            )

        # Evaluate pipeline 1
        print("Evaluating Pipeline 1...")
        result1 = evaluate(
            query_engine=self.pipeline1.get_query_engine(),
            metrics=self.metrics,
            dataset=self.evaluation_questions,
        )

        # Evaluate pipeline 2
        print("Evaluating Pipeline 2...")
        result2 = evaluate(
            query_engine=self.pipeline2.get_query_engine(),
            metrics=self.metrics,
            dataset=self.evaluation_questions,
        )

        # Get results as pandas DataFrame
        result1_df = result1.to_pandas()
        result2_df = result2.to_pandas()

        # Add empty context_relevancy and hallucinations column to both DataFrames as float
        result1_df["context_relevancy"] = pd.Series(dtype=float)
        result2_df["context_relevancy"] = pd.Series(dtype=float)
        result1_df["hallucination"] = pd.Series(dtype=float)
        result2_df["hallucination"] = pd.Series(dtype=float)

        # Calculate hallucinations
        result1_df["hallucination"] = 1 - result1_df["faithfulness"]
        result2_df["hallucination"] = 1 - result2_df["faithfulness"]

        # Calculate context relevancy (f1 score)
        result1_df["context_relevancy"] = np.where(
            (result1_df["context_precision"] + result1_df["context_recall"]) > 0,
            2
            * (result1_df["context_precision"] * result1_df["context_recall"])
            / (result1_df["context_precision"] + result1_df["context_recall"]),
            0,
        )

        result2_df["context_relevancy"] = np.where(
            (result2_df["context_precision"] + result2_df["context_recall"]) > 0,
            2
            * (result2_df["context_precision"] * result2_df["context_recall"])
            / (result2_df["context_precision"] + result2_df["context_recall"]),
            0,
        )

        # Store the results with the new metric
        self.results["pipeline1"] = result1_df
        self.results["pipeline2"] = result2_df
        self.results["pipeline1"].to_json(
            os.path.join(persist_path, "pipeline1_results.json")
        )
        self.results["pipeline2"].to_json(
            os.path.join(persist_path, "pipeline2_results.json")
        )

        return self.results

    def get_summary_statistics(self):
        """Get summary statistics of the evaluation results."""
        if self.results["pipeline1"].empty or self.results["pipeline2"].empty:
            return "Run evaluation first by calling run_evaluation()"

        # Extract metrics columns
        metric_columns = [
            col
            for col in self.results["pipeline1"].columns
            if col
            not in [
                "user_input",
                "retrieved_contexts",
                "response",
                "reference",
                "reference_contexts",
            ]
        ]

        # Calculate mean scores for each metric
        pipeline1_means = self.results["pipeline1"][metric_columns].mean()
        pipeline2_means = self.results["pipeline2"][metric_columns].mean()

        # Create summary DataFrame
        summary = pd.DataFrame(
            {
                "Pipeline 1": pipeline1_means,
                "Pipeline 2": pipeline2_means,
                "Difference": pipeline1_means - pipeline2_means,
            }
        )

        # Add winning pipeline column
        winners = []
        for idx, metric in enumerate(summary.index):
            diff = summary["Difference"][idx]
            if metric == "hallucination":
                # For hallucination, lower is better
                winners.append(
                    "Pipeline 1" if diff < 0 else "Pipeline 2" if diff > 0 else "Tie"
                )
            else:
                # For all other metrics, higher is better
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

        # Get metrics
        metrics = list(summary.index)

        # Get scores
        pipeline1_scores = [summary["Pipeline 1"][metric] for metric in metrics]
        pipeline2_scores = [summary["Pipeline 2"][metric] for metric in metrics]

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
        plt.title("RAGAS Evaluation Comparison", size=15)

        # Save if output file is specified
        if output_file:
            plt.savefig(
                os.path.join(persist_path, output_file), dpi=300, bbox_inches="tight"
            )

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Create a bar chart for side-by-side comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(
            x - width / 2,
            [summary["Pipeline 1"][m] for m in metrics],
            width,
            label="Pipeline 1",
        )
        ax.bar(
            x + width / 2,
            [summary["Pipeline 2"][m] for m in metrics],
            width,
            label="Pipeline 2",
        )

        ax.set_ylabel("Score")
        ax.set_title("RAGAS Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.legend()

        plt.tight_layout()

        if output_file:
            base, ext = os.path.splitext(output_file)
            bar_output = f"{base}_bar{ext}"
            plt.savefig(
                os.path.join(persist_path, bar_output), dpi=300, bbox_inches="tight"
            )

        plt.show()

        return summary

    async def evaluate(
        self,
        num_questions: int = 20,
        output_file="ragas_evaluation_results.pdf",
    ) -> pd.DataFrame:
        """
        Run a comprehensive evaluation of the two configured RAG pipelines using RAGAS metrics.

        Args:
            num_questions: Number of evaluation questions to generate and use
            output_file: Path to save visualization results

        Returns:
            pd.DataFrame: Summary statistics comparing the performance of both pipelines
        """

        print("Running RAGAS evaluation...")
        await self.run_evaluation(num_questions)

        print("\nSummary Statistics:")
        summary = self.get_summary_statistics()
        print(summary)

        print("\nGenerating visualizations...")
        self.visualize_results(output_file=output_file)

        return summary.to_dict()
