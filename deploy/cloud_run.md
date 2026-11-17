# WorkspaceRAG Cloud Deployment Guide

This guide details how to deploy the **WorkspaceRAG** stack (FastAPI Backend, PostgreSQL + pgvector, and Streamlit Frontend) to **Google Cloud Run** and **Railway**.

---

## 1. Option A: 1-Click Railway Deployment (Fastest)

Railway natively provides hosted PostgreSQL 16 with the `pgvector` extension enabled by default.

### Steps:
1. Fork or push this repository to GitHub.
2. Go to [Railway Dashboard](https://railway.app/new).
3. Click **Deploy from GitHub repo** and select `Enterprise-CRAG`.
4. In Railway project settings, click **+ New** $\rightarrow$ **Database** $\rightarrow$ **Add PostgreSQL**.
5. Enable `pgvector` in the PostgreSQL database:
   - Connect via Railway Query Console and run:
     ```sql
     CREATE EXTENSION IF NOT EXISTS vector;
     ```
6. Set the following Environment Variables on your FastAPI service:
   - `DATABASE_URL`: `${{Postgres.DATABASE_URL}}`
   - `GEMINI_API_KEY`: Your Google AI Studio API key
   - `COHERE_API_KEY`: Your Cohere API key
   - `LANGFUSE_SECRET_KEY`: Your Langfuse secret key
   - `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key
   - `LANGFUSE_HOST`: `https://cloud.langfuse.com`
7. Railway uses [`railway.json`](file:///c:/Users/KIIT/OneDrive/Desktop/Dev/workspace-rag/railway.json) to automatically build the Dockerfile and configure `/api/v1/query/health` as the zero-downtime healthcheck endpoint.

---

## 2. Option B: Google Cloud Run (Serverless Production)

Google Cloud Run scales containers from 0 to $N$ with built-in HTTPS and automatic TLS termination.

### Prerequisites:
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- A Google Cloud Project with Cloud Run, Cloud Build, and Secret Manager enabled

### Deployment Script:

```bash
# 1. Set Google Cloud project variables
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/workspace-rag-api:latest"

# 2. Build container image using Cloud Build
gcloud builds submit --tag ${IMAGE_NAME} .

# 3. Deploy FastAPI backend to Cloud Run
gcloud run deploy workspace-rag-api \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "LANGFUSE_HOST=https://cloud.langfuse.com" \
  --set-secrets "DATABASE_URL=workspace_rag_db_url:latest,GEMINI_API_KEY=gemini_api_key:latest,COHERE_API_KEY=cohere_api_key:latest,LANGFUSE_SECRET_KEY=langfuse_secret_key:latest,LANGFUSE_PUBLIC_KEY=langfuse_public_key:latest"

# 4. Deploy Streamlit frontend to Cloud Run
API_URL=$(gcloud run services describe workspace-rag-api --platform managed --region ${REGION} --format 'value(status.url)')

gcloud run deploy workspace-rag-ui \
  --image ${IMAGE_NAME} \
  --command "streamlit" \
  --args "run,streamlit_app.py,--server.port=8080,--server.address=0.0.0.0" \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --set-env-vars "API_BASE=${API_URL}"
```

---

## 3. Database Migration & Table Setup

When deploying to a fresh database instance (Cloud SQL, Supabase, Neon, or Railway):
1. Ensure the vector extension is loaded:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. The FastAPI backend automatically initializes the `documents`, `document_chunks`, and `rag_query_cache` tables on startup during its lifespan context manager.
