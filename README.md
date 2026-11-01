# Intelligent Document Research Assistant

Intelligent Document Research Assistant is a Streamlit application for asking questions across uploaded documents. It combines semantic retrieval powered by Amazon Bedrock embeddings with keyword retrieval, then uses an AWS Bedrock chat model to answer with document context. When the available document context or model knowledge is insufficient, it can fall back to web search through Tavily.

## Features

- Upload PDF, DOCX, TXT, and Markdown files.
- Split uploaded documents into overlapping text chunks and build an in-memory FAISS vector index.
- Retrieve relevant passages with a hybrid search strategy:
  - Dense semantic search using Amazon Titan Text Embeddings on Amazon Bedrock.
  - Sparse keyword search using BM25.
  - Reciprocal-rank fusion to combine and rank both result sets.
- Grade retrieved passages for relevance before generating an answer.
- Check generated responses for grounding and usefulness, with a limited corrective generation loop.
- Use the Bedrock chat model's general knowledge and optionally Tavily web search when documents do not provide an answer.
- Keep a short conversation history in a local `chat_memory.json` file.
- Clear the indexed documents or conversation from the sidebar.

## How it works

1. Upload documents and select **Index documents**.
2. The app extracts text, splits it into chunks, embeds the chunks with the configured Bedrock embedding model, and builds the dense and BM25 indexes.
3. When you ask a question, the app rewrites it using recent conversation history when appropriate, then searches the document indexes.
4. A Bedrock chat model grades the retrieved passages and generates an answer from relevant context.
5. If no useful document context is found, the assistant tries its general knowledge. If it is not confident or the question needs current information, Tavily web search is used as a fallback.
6. The answer, source, processing steps, and available references are shown in the chat.

The FAISS index is held in application memory and is cleared when the app process restarts. Conversation history is stored locally in `chat_memory.json`.

## Requirements

- Python 3.10 or later.
- AWS credentials configured for the AWS SDK, with access to Amazon Bedrock in the selected region.
- Bedrock model access enabled for the configured chat and embedding models.
- A Tavily API key for web-search fallback.

The app uses the `amazon.titan-embed-text-v2:0` model by default for embeddings. Choose a Bedrock chat model supported by your AWS account and region for `BEDROCK_MODEL_ID`.

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set the model IDs and Tavily key:

```powershell
Copy-Item .env.example .env
```

Configure these values in `.env`:

| Variable | Purpose |
| --- | --- |
| `AWS_REGION` | AWS region where your Bedrock models are enabled. |
| `BEDROCK_MODEL_ID` | Bedrock chat model ID used for answering, grading, and rewriting. |
| `BEDROCK_EMBEDDING_MODEL_ID` | Bedrock embedding model ID. Defaults to `amazon.titan-embed-text-v2:0`. |
| `TAVILY_API_KEY` | API key used for web-search fallback. |

AWS authentication is provided by the standard AWS credential chain (for example, an AWS profile, environment credentials, or an attached IAM role). Do not put AWS access keys in the README or commit them to source control. The AWS identity must be allowed to invoke the selected Bedrock models and access to those models must be enabled in the AWS account.

### 4. Start the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, upload documents in the sidebar, and select **Index documents** before asking questions about them.

## Project files

| File | Description |
| --- | --- |
| `app.py` | Streamlit user interface, document upload, and chat interaction. |
| `rag.py` | Document loading, chunking, Bedrock embeddings, hybrid retrieval, and ranking. |
| `agent.py` | LangGraph workflow for retrieval, grading, answer generation, reflection, and web fallback. |
| `memory.py` | Local JSON-backed conversation history. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for application configuration. |

## Notes

- Uploaded document contents and generated embeddings are processed using the AWS Bedrock model configured for embeddings. Review applicable AWS data handling and regional requirements before using sensitive documents.
- Web search requires a working `TAVILY_API_KEY`; document retrieval and Bedrock answers are otherwise independent of Tavily.
- This prototype keeps indexes in memory and does not provide persistent vector storage, user authentication, or multi-user isolation.
