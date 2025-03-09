import os
import asyncio
import argparse
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.pipelines.agentic_pipeline import AgenticRAGPipeline
from src.pipelines.standard_pipeline import StandardRAGPipeline
from src.utils.utils import get_pdf_paths


def main_rag_cli():

    parser = argparse.ArgumentParser(description="Test the RAGPiepline")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="storage/raw",
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--embedder_type",
        type=str,
        default="huggingface",
        help="Embedder type (e.g., openai, cohere, huggingface)",
    )
    parser.add_argument(
        "--embedder_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name",
    )
    parser.add_argument(
        "--retriever_type",
        type=str,
        default="vector",
        help="Retriever type (e.g., vector, hybrid)",
    )
    parser.add_argument(
        "--generator_llm_type",
        type=str,
        default="openai",
        help="LLM type for generation (e.g., openai, mistral)",
    )

    parser.add_argument(
        "--rag_pipeline_type",
        type=str,
        default="standard",
        help="RAG Pipeline type (e.g., standard, agentic)",
    )

    args = parser.parse_args()

    # Find PDF files in the specified directory
    pdf_files = []
    if os.path.exists(args.pdf_dir):
        for file in os.listdir(args.pdf_dir):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(args.pdf_dir, file))

    if not pdf_files:
        print(f"No PDF files found in {args.pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")

    # Initialize the pipeline
    if args.rag_pipeline_type == "standard":
        pipeline = StandardRAGPipeline(
            embedder_type=args.embedder_type,
            embedder_model=args.embedder_model,
            retriever_type=args.retriever_type,
            generator_llm_type=args.generator_llm_type,
        )
    elif args.rag_pipeline_type == "agentic":
        pipeline = AgenticRAGPipeline(
            embedder_type=args.embedder_type,
            embedder_model=args.embedder_model,
            retriever_type=args.retriever_type,
            generator_llm_type=args.generator_llm_type,
        )

    # Load documents
    print("Loading documents...")
    pipeline.load_pipeline(pdf_files)
    print("Pipeline loaded successfully")

    # Interactive questioning
    print("\nEnter your questions (type 'exit' to quit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "exit":
            break

        response = pipeline.query(question)
        print(f"\nAnswer: {response}")


async def main_rag_evaluate():
    from src.pipelines.standard_pipeline import StandardRAGPipeline
    from src.evaluation.llama_evaluator import RAGEvaluator
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.openai import OpenAI

    # Load data source file paths
    data_path = "storage/raw"
    pdf_files = get_pdf_paths(data_path)

    # Create LLM
    # llm = Anthropic(model="claude-3-7-sonnet-latest", temperature=0.1)
    llm = OpenAI(model="gpt-4o", temperature=0.1)

    # Create two different index configurations (pipelines)
    # Pipeline 1: Basic vector index with OpenAI LLM
    pipeline_1 = StandardRAGPipeline(
        embedder_type="huggingface",
        embedder_model=None,
        retriever_type="vector",
        generator_llm_type="openai",
    )

    # Pipeline 2: Basic vector index with Mistral LLM
    pipeline_2 = AgenticRAGPipeline(
        embedder_type="huggingface",
        embedder_model=None,
        retriever_type="hybrid",
        generator_llm_type="openai",
    )

    # Load the pipelines
    print("Loading documents : ", pdf_files)
    await pipeline_1.load_pipeline(pdf_files)
    await pipeline_2.load_pipeline(pdf_files)
    print("Pipeline loaded successfully")

    # Evaluate the pipelines
    print("Evaluating pipelines...")
    # evaluator = RAGEvaluator(
    #     pipeline1=pipeline_1, pipeline2=pipeline_2, llm=llm, embed_model="huggingface"
    # )
    evaluator = RAGASEvaluator(
        pipeline1=pipeline_1, pipeline2=pipeline_2, llm=llm, embed_model="huggingface"
    )
    await evaluator.evaluate()


if __name__ == "__main__":
    asyncio.run(main_rag_evaluate())
    # main_rag_cli()
