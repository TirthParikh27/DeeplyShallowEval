# DeeplyShallowEval

A modular framework for creating, comparing, and evaluating RAG (Retrieval-Augmented Generation) pipelines. This tool allows users to build different RAG configurations and benchmark them against each other using popular evaluation frameworks.

## Features

- **Modular Pipeline Construction**: Easily build and configure RAG pipelines with different components
- **Multiple Evaluation Frameworks**: Support for RAGAS and LlamaIndex evaluation methodologies
- **Multiple LLM Providers**: Compatible with OpenAI and Anthropic models
- **PDF Document Processing**: Built-in support for processing PDF documents
- **API-First Design**: Simple REST API to create and evaluate pipelines
- **Comparative Analysis**: Benchmark two different RAG pipelines against each other

## Installation and Setup

### Prerequisites

- Python 3.12+
- pip
- Docker and Docker Compose (for containerized deployment)
- Git

### Setting Up for Local Development

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/DeeplyShallowEval.git
   cd DeeplyShallowEval
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating a `.env` file:

   ```
   OPENAI_API_KEY=your_openai_key
   LLAMA_CLOUD_API_KEY=your_llama_key
   COHERE_API_KEY=your_cohere_key
   ```

5. Run the application:
   ```bash
   cd api
   python main.py
   ```

### Setting Up with Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/DeeplyShallowEval.git
   cd DeeplyShallowEval
   ```

2. Create a `.env` file with your API keys:

   ```
   OPENAI_API_KEY=your_openai_key
   LLAMA_CLOUD_API_KEY=your_llama_key
   COHERE_API_KEY=your_cohere_key
   ```

3. Build and start the Docker container:

   ```bash
   docker-compose up -d --build
   ```

4. The API will be available at `http://localhost:8505`

5. To stop the container:

   ```bash
   docker-compose down
   ```

6. To view logs:
   ```bash
   docker-compose logs -f
   ```

### Persistent Storage

When using Docker,the `./storage` directory will be mounted inside the container, ensuring data persistence between container restarts. Add the PDF documents for evaluation under `./storage/raw` folder.

## Usage

### Starting the API Server

```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### POST /evaluation/eval_rag

Evaluate and compare two RAG pipelines.

Example request:

```json
{
  "pipeline1_config": {
    "pipeline_type": "standard",
    "embedder_type": "huggingface",
    "embedder_model": "BAAI/bge-large-en-v1.5",
    "retriever_type": "vector",
    "generator_llm_type": "openai"
  },
  "pipeline2_config": {
    "pipeline_type": "standard",
    "embedder_type": "huggingface",
    "embedder_model": "BAAI/bge-large-en-v1.5",
    "retriever_type": "hybrid",
    "generator_llm_type": "openai"
  },
  "evaluator_type": "ragas",
  "llm_provider": "openai",
  "llm_model": "gpt-4o"
}
```

## Project Structure

```
DeeplyShallowEval/
├── api/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── routers/
│   │   └── eval_router.py # API routes for evaluation
├── rag/
│   ├── evaluation/
│   │   ├── ragas_evaluator.py  # RAGAS evaluation implementation
│   │   └── llama_evaluator.py  # LlamaIndex evaluation implementation
│   ├── utils/
│   │   └── utils.py     # Utility functions for RAG
├── schemas/
│   └── eval_schemas.py  # Pydantic models for API
├── storage/
│   └── raw/            # Storage for PDF documents
├── utils/
│   └── utils.py        # General utility functions
├── .env                # Environment variables
└── requirements.txt    # Python dependencies
```

## Evaluation Metrics

DeeplyShallowEval supports various evaluation metrics:

### RAGAS Metrics

- Answer Relevancy
- Faithfulness
- Context Precision
- Context Recall

### LlamaIndex Metrics

- Correctness
- Relevance
- Coherence
- Groundedness

## Configuration Options

### LLM Providers

- OpenAI: gpt-3.5-turbo, gpt-4
- Anthropic: claude-2, claude-instant

### Embedding Models

- OpenAI
- Hugging Face
- Cohere

### Retrieval Methods

- Vector Store
- Hybrid Search

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
