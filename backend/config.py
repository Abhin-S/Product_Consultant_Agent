from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ENVIRONMENT: str = "development"
    AUTH_MODE: str = "google_sso"

    # Database
    # Dev example: sqlite+aiosqlite:///./dev.db
    # Prod example: postgresql+asyncpg://user:pass@host/db
    DATABASE_URL: str = "sqlite+aiosqlite:///./dev.db"

    # ChromaDB
    CHROMA_DB_PATH: str = "./chroma_db"
    DOCS_DIR: str = "./docs"
    PARENT_STORE_PATH: str = "./processed_data/parent_chunks.json"

    # Google AI Studio — Gemma 4
    GOOGLE_API_KEY: str
    GEMMA_MODEL_NAME: str = "gemma-4-it"
    LLM_FALLBACK_MODEL_NAME: str = "gemma-3-12b-it"

    # Google SSO (OAuth)
    GOOGLE_OAUTH_CLIENT_ID: str = ""
    GOOGLE_OAUTH_CLIENT_SECRET: str = ""
    GOOGLE_OAUTH_REDIRECT_URI: str = ""
    GOOGLE_OAUTH_SCOPES: str = "openid email profile"
    AUTH_COOKIE_NAME: str = "pcai_access_token"
    AUTH_COOKIE_SECURE: bool = False
    AUTH_COOKIE_SAMESITE: str = "lax"

    # RAG
    CONFIDENCE_THRESHOLD: float = 0.45
    TOP_K_DEFAULT: int = 5
    MAX_CONTEXT_TOKENS: int = 3000
    LLM_MAX_RETRIES: int = 3
    MODEL_REQUEST_TIMEOUT_SECONDS: int = 30
    BYPASS_LLM_CALLS: bool = False
    ENABLE_QUERY_EXPANSION: bool = True
    ENABLE_RELEVANCE_GRADING: bool = True
    MULTI_QUERY_COUNT: int = 3
    RRF_K: int = 60
    CRAG_MIN_RELEVANT_DOCS: int = 2
    ENABLE_GROUNDING_CHECK: bool = True
    PARENT_CHUNK_SIZE: int = 1800
    PARENT_CHUNK_OVERLAP: int = 180
    CHILD_CHUNK_SIZE: int = 500
    CHILD_CHUNK_OVERLAP: int = 50
    MAX_PARENT_CONTEXT_CHUNKS: int = 3

    # News
    NEWS_API_KEY: str = ""
    GNEWS_API_KEY: str = ""
    NEWS_RELEVANCE_THRESHOLD: float = 0.35
    NEWS_MAX_AGE_DAYS: int = 30

    # Encryption
    FERNET_KEY: str

    # OAuth stubs (for future v2)
    NOTION_CLIENT_ID: str = ""
    NOTION_CLIENT_SECRET: str = ""
    NOTION_REDIRECT_URI: str = ""
    JIRA_CLIENT_ID: str = ""
    JIRA_CLIENT_SECRET: str = ""
    JIRA_REDIRECT_URI: str = ""

    # Evaluation
    EVAL_SAMPLE_RATE: int = 10
    RAGAS_TIMEOUT_SECONDS: int = 30
    RAGAS_MAX_OUTPUT_TOKENS: int = 0
    RAGAS_MAX_CONTEXT_DOCS: int = 4
    RAGAS_CONTEXT_DOC_CHAR_LIMIT: int = 1800
    RAGAS_ANSWER_CHAR_LIMIT: int = 3000

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()