# DeeplyShallowEval

A modular framework for creating, comparing, and evaluating RAG (Retrieval-Augmented Generation) pipelines. This tool allows users to build different RAG configurations and benchmark them against each other using popular evaluation frameworks.

## Features

- **Modular Pipeline Construction**: Easily build and configure RAG pipelines with different components
- **Evaluation Framework**: Support for RAGAS evealutation framework
- **Multiple LLM Providers**: Compatible with OpenAI , Anthropic ad Mistral models
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

4. Set up environment variables by creating a `.env` file in the root directory:

   ```
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    MISTRAL_API_KEY=your_mistral_api_key_here
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
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

Create, evaluate and compare two RAG pipelines.

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
├── api/                          # API layer
│   ├── rag/                      # RAG implementation
│   │   ├── data/                 # Data processor and parser for RAG
│   │   ├── embeddings/           # Vector embedding models
│   │   ├── evaluation/           # Evaluation tools and metrics
│   │   ├── generation/           # Text generation components
│   │   ├── pipelines/            # RAG pipelines (standard , agentic)
│   │   ├── retrieval/            # Retrieval components
│   │   ├── utils/                # Utility functions for RAG
│   ├── routers/                  # API route definitions
│   ├── schemas/                  # Data schemas
│   ├── utils/                    # General utility functions
│   ├── config.py                 # Configuration settings
│   └── main.py                   # API entry point
├── storage/                      # Persistant Data storage (pdfs in ./storage/raw)
├── .env                          # Environment variables
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker configuration
```

## Evaluation Metrics

DeeplyShallowEval supports various evaluation metrics:

### Metrics

- Answer Relevancy
- Faithfulness
- Context Precision
- Context Recall
- Hallucination
- Context Relevancy (f-score)

## Configuration Options

### Evaluator LLMs

- OpenAI: gpt-4o
- Anthropic: claude-3.7-sonnet

### LLM Providers

- OpenAI: gpt4o-mini
- Anthropic: claude-3.5-haiku
- Mistral: mistral-small-latest

### Embedding Models

- OpenAI
- Hugging Face
- Cohere

### Retrieval Methods

- Vector Store
- Hybrid Search (Vector + BM25)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License Version 2.0 - see the LICENSE file for details.
