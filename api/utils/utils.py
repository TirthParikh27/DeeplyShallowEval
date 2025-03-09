from rag.pipelines.standard_pipeline import StandardRAGPipeline
from rag.pipelines.agentic_pipeline import AgenticRAGPipeline
from schemas.eval_schemas import PipelineConfig


def create_pipeline(config: PipelineConfig):
    """Helper function to create a pipeline from configuration"""
    if config.pipeline_type.lower() == "standard":
        return StandardRAGPipeline(
            embedder_type=config.embedder_type,
            embedder_model=config.embedder_model,
            retriever_type=config.retriever_type,
            generator_llm_type=config.generator_llm_type,
        )
    elif config.pipeline_type.lower() == "agentic":
        return AgenticRAGPipeline(
            embedder_type=config.embedder_type,
            embedder_model=config.embedder_model,
            retriever_type=config.retriever_type,
            generator_llm_type=config.generator_llm_type,
        )
    else:
        raise ValueError(f"Unsupported pipeline type: {config.pipeline_type}")
