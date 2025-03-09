from fastapi import FastAPI
from routers.eval_router import router as eval_router
import uvicorn

app = FastAPI(
    title="DeeplyShallowEval API",
    description="API for evaluating RAG pipelines",
    version="0.1.0",
)

app.include_router(eval_router)

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
