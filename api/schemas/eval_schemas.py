from pydantic import BaseModel
from typing import Any, Dict, Optional


class PipelineConfig(BaseModel):
    pipeline_type: str  # "standard" or "agentic"
    embedder_type: str
    embedder_model: Optional[str] = None
    retriever_type: str
    generator_llm_type: str


class EvaluationRequest(BaseModel):
    pipeline1_config: PipelineConfig
    pipeline2_config: PipelineConfig
    evaluator_type: str = "ragas"  # "ragas" or "llama"
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4o"


class EvaluationResponse(BaseModel):
    results: Dict[str, Any]
