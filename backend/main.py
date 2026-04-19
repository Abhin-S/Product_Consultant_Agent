import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analyze import router as analyze_router
from api.routes.eval_admin import router as eval_admin_router
from api.routes.execute import router as execute_router
from api.routes.ingest import router as ingest_router
from api.routes.sessions import router as sessions_router
from auth.bootstrap import ensure_default_admin
from auth.router import router as auth_router
from config import settings
from evaluation.ragas_evaluator import log_ragas_version
from ingestion.embedder import load_embedder_model
from integrations.router import router as integrations_router
from retrieval.parent_store import warm_parent_store
from retrieval.vector_store import get_collection, purge_expired_dynamic_documents


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Ensuring default admin (if configured)")
    await ensure_default_admin()

    logger.info("Loading sentence-transformer singleton model")
    load_embedder_model()

    logger.info("Loading parent chunk store")
    warm_parent_store()

    logger.info("Ensuring ChromaDB collection exists")
    get_collection()

    logger.info("Purging expired dynamic docs from ChromaDB")
    purge_expired_dynamic_documents(hours=24)

    log_ragas_version()

    yield


app = FastAPI(title="AI Product Consultant Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "gemma_model": settings.GEMMA_MODEL_NAME}


app.include_router(auth_router)
app.include_router(integrations_router)
app.include_router(ingest_router)
app.include_router(analyze_router)
app.include_router(execute_router)
app.include_router(sessions_router)
app.include_router(eval_admin_router)