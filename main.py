import os
import argparse
from src.pipelines.agentic_pipeline import AgenticRAGPipeline
from src.pipelines.standard_pipeline import StandardRAGPipeline
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings


def main():

    parser = argparse.ArgumentParser(description="Test the RAGPiepline")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="data/raw",
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

    if args.generator_llm_type == "anthropic":
        llm = Anthropic(model="claude-3-5-sonnet-latest")
        Settings.tokenizer = llm.tokenizer.encode

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

        response = pipeline.answer_question(question)
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
