from fastapi import APIRouter, HTTPException, Body
import os

from schemas.eval_schemas import (
    EvaluationRequest,
    EvaluationResponse,
)
from rag.evaluation.ragas_evaluator import RAGASEvaluator
from rag.evaluation.llama_evaluator import RAGEvaluator
from rag.utils.utils import get_pdf_paths
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

from utils.utils import create_pipeline

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/eval_rag", response_model=EvaluationResponse)
async def evaluate_rag_pipelines(request: EvaluationRequest = Body(...)):
    try:
        # Validate data path and get PDF files
        data_path = os.path.join("storage", "raw")
        if not os.path.exists(data_path):
            raise HTTPException(
                status_code=404, detail=f"Data path {data_path} not found"
            )

        pdf_files = get_pdf_paths(data_path)
        if not pdf_files:
            raise HTTPException(
                status_code=404, detail=f"No PDF files found in {data_path}"
            )

        # Create evaluator LLM based on configuration
        if request.llm_provider == "openai":
            llm = OpenAI(model=request.llm_model, temperature=0.1)
        elif request.llm_provider == "anthropic":
            llm = Anthropic(model=request.llm_model, temperature=0.1)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported LLM provider: {request.llm_provider}",
            )

        # Create Pipeline 1
        pipeline1 = create_pipeline(request.pipeline1_config)

        # Create Pipeline 2
        pipeline2 = create_pipeline(request.pipeline2_config)

        # Load the pipelines
        await pipeline1.load_pipeline(pdf_files)
        await pipeline2.load_pipeline(pdf_files)

        # Create appropriate evaluator
        if request.evaluator_type.lower() == "ragas":
            evaluator = RAGASEvaluator(
                pipeline1=pipeline1,
                pipeline2=pipeline2,
                llm=llm,
                embed_model="huggingface",
            )
        elif request.evaluator_type.lower() == "llama":
            evaluator = RAGEvaluator(
                pipeline1=pipeline1,
                pipeline2=pipeline2,
                llm=llm,
                embed_model="huggingface",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported evaluator type: {request.evaluator_type}",
            )

        # Run evaluation
        results = await evaluator.evaluate()

        # Return results
        return EvaluationResponse(
            results=results,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
