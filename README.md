# AI Product Consultant Agent - Local Setup

This project is split into independent runtime environments:

- `backend/` uses Python dependencies from `backend/requirements.txt`
- `frontend/` uses Node dependencies from `frontend/package.json`

## Backend (Python via uv)

Use Python 3.11 for backend. This avoids Windows build-tool issues with `chroma-hnswlib`.

Run these commands in a terminal:

```powershell
cd backend
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
```

Fill the required values in `.env` (at least `SECRET_KEY`, `GOOGLE_API_KEY`, `FERNET_KEY`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`).

Start backend:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

## Frontend (npm)

Run these commands in a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

Yes, for frontend your understanding is correct: `npm install` then `npm run dev` is enough for local development.

## Notes

- Chroma documents/chunks are created only after you run ingestion once.
- Re-running ingestion now refreshes chunks for files included in the run (same source path), so updated files do not require manual Chroma deletion.
- If Swagger auth is inconvenient with Google SSO-only login, run local ingestion from terminal: `& .\.venv\Scripts\python.exe ingest_local.py` (from `backend/`).
- The root `.venv/` is not required for this setup path. The recommended Python environment is `backend/.venv`.
- Authentication uses Google SSO only (`AUTH_MODE=google_sso`).
- Auth tokens are stored in HttpOnly cookies (not in URL query params or browser localStorage).
- Role separation is removed; routes are authenticated per user and session data remains scoped by `user_id`.

## API Keys You Need (and How to Get Them)

Required for local run:

1. `GOOGLE_API_KEY` (Gemma model access)
	- Get it from Google AI Studio: https://aistudio.google.com/app/apikey
	- Create a key, then set it in `backend/.env`.

2. `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` (Google SSO)
	- Go to Google Cloud Console: APIs & Services -> Credentials.
	- Create an OAuth Client ID of type Web application.
	- Add Authorized redirect URI: `http://localhost:8000/auth/google/callback`
	- Add Authorized JavaScript origin: `http://localhost:3000`
	- Copy client ID and secret into `backend/.env`.

3. `SECRET_KEY` (JWT signing)
	- Generate with:
	- `python -c "import secrets; print(secrets.token_urlsafe(48))"`

4. `FERNET_KEY` (encryption key for stored integration secrets)
	- Generate with:
	- `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`

Optional keys:

1. `NEWS_API_KEY`
	- Source: https://newsapi.org
	- Needed only for news/web fallback enrichment.

2. `GNEWS_API_KEY`
	- Source: https://gnews.io
	- Optional alternative news provider.

Notion and Jira credentials (currently optional / planned OAuth):

1. `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET`, `NOTION_REDIRECT_URI`
	- Current project status: Notion OAuth route is working and are implemented.
	- Expected callback route in current backend router: `http://localhost:8000/integrations/notion/callback`
	- How to acquire:
	- Go to Notion integrations portal: https://www.notion.so/my-integrations
	- Create a new integration/app and enable OAuth (public integration flow).
	- Copy Client ID and Client Secret into `backend/.env`.
	- Set redirect URI to the callback route above.
	- Explicit uncertainty: Notion's portal labels and OAuth setup screens can change. If you only create an "internal" integration, you may get an integration token instead of OAuth client credentials.

2. `JIRA_CLIENT_ID`, `JIRA_CLIENT_SECRET`, `JIRA_REDIRECT_URI`
	- Current project status: Jira OAuth route is a stub, so these are not required to run the app today.
	- Expected callback route in current backend router: `http://localhost:8000/integrations/jira/callback`
	- How to acquire:
	- Go to Atlassian Developer Console: https://developer.atlassian.com/console/myapps/
	- Create an OAuth 2.0 (3LO) app.
	- Add callback URL as the Jira callback route above.
	- Copy Client ID and Client Secret into `backend/.env`.
	- Explicit uncertainty: exact Atlassian console wording and required scopes depend on your Jira Cloud site and the actions you plan to support.

## Next Steps After Ingestion Is Done

If you already ran ingestion successfully, do this next:

1. Start backend (from `backend/`):

```powershell
& .\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

2. Start frontend (from `frontend/`):

```powershell
npm run dev
```

3. Login at `http://localhost:3000/login` using "Continue with Google".
4. Go to Analyze page and submit your first startup idea.
5. Re-run ingestion only when corpus changes:
	- Standard re-index (recommended for added/updated files): `& .\.venv\Scripts\python.exe ingest_local.py`
	- Full rebuild (required for deleted/renamed files or schema/chunking changes): `& .\.venv\Scripts\python.exe ingest_local.py --rebuild`

## Re-ingestion Playbook (After Corpus Expansion)

If you added more case-study files (including subfolders), follow this:

1. From `backend/`, activate environment and sync dependencies:

```powershell
& .\.venv\Scripts\Activate.ps1
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

2. Run standard re-index:

```powershell
& .\.venv\Scripts\python.exe ingest_local.py
```

Optional: run against a specific directory instead of `DOCS_DIR` from `.env`:

```powershell
& .\.venv\Scripts\python.exe ingest_local.py --docs-dir "D:\path\to\docs"
```

What this does now:
- Reloads all supported documents from `DOCS_DIR` recursively.
- Extracts PDFs with table-aware parsing (page text + Markdown tables).
- Extracts DOCX body text and tables, preserving table structure in Markdown.
- Skips legacy `.doc` files (convert them to `.docx` for reliable extraction).
- Rebuilds parent/child chunks for loaded sources.
- Adds source metadata (relative path, topic, subtopic, section/page signals) to chunks.
- Automatically batches Chroma upserts to respect Chroma per-request limits while still ingesting the full corpus.
- Replaces existing local chunks for those sources in Chroma.
- Replaces parent-store entries for those sources.

Should you delete existing chunks / Chroma DB first?
- If you only added or edited files: No. Run standard re-index only.
- If you deleted files and want removed content gone from retrieval: Use `--rebuild`.
- If you renamed/moved many files (path identity changed): Use `--rebuild`.
- If you changed chunk-size settings (`PARENT_*` / `CHILD_*`) or ingestion logic: Use `--rebuild`.

When you should run full rebuild:
- You deleted files from corpus and want removed content fully gone from index.
- You renamed/moved many files (source path identity changed).
- You changed chunk-size settings (`PARENT_*`/`CHILD_*`) and want a clean index.
- You changed ingestion logic and want to eliminate any stale artifacts.

Full rebuild command:

```powershell
& .\.venv\Scripts\python.exe ingest_local.py --rebuild
```

Do you need to manually delete Chroma DB files?
- Usually no. Prefer `--rebuild`; it clears the collection and parent store safely.
- Manual deletion should only be a last resort if Chroma state is corrupted.

## Evaluation Fallback (When RAGAS Fails)

The system keeps RAGAS as the primary Tier-2 evaluator.
If RAGAS fails, it automatically runs a traditional benchmark fallback.

Benchmark file:
- Path: `backend/evaluation/benchmark_queries.json`
- Required keys per item:
	- `query`
	- `document_id`
	- `reference_answer`

Traditional fallback metrics:
- Retrieval: `Recall@k`, `MAP@k`
- Generation: `ROUGE-L F1`, `BERTScore F1`

How fallback status appears:
- `ragas_eval_status` is set to `fallback_completed`
- API payload includes:
	- `evaluation_mode: traditional_fallback`
	- `evaluation_notice` (explicitly states predefined benchmark queries were used)
	- `traditional_metrics`

Important behavior:
- Benchmark generation calls run with timeout override disabled (`timeout_override_seconds=0`) for fallback evaluation.
- The fallback metrics are benchmark-level diagnostics, not a direct score for the user's exact question.

## First Run Checklist

1. Create and activate backend venv in `backend/`.
2. Install backend dependencies from `backend/requirements.txt`.
3. Copy `backend/.env.example` to `backend/.env` and fill required keys.
4. In Google Cloud Console, create a Web OAuth client and add the redirect URI `http://localhost:8000/auth/google/callback`.
5. Run Alembic migrations from `backend/`:

```powershell
& .\.venv\Scripts\python.exe -m alembic upgrade head
```

6. Start backend:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

7. In `frontend/`, run `npm install`, copy `.env.local.example` to `.env.local`, then run `npm run dev`.
8. Use "Continue with Google" on `/login`.
9. Build the local vector index (from `backend/`) if not already done:

```powershell
& .\.venv\Scripts\python.exe ingest_local.py
```

10. Optional full rebuild (clears existing local index, then ingests again):

```powershell
& .\.venv\Scripts\python.exe ingest_local.py --rebuild
```

11. You can still trigger ingestion from `http://localhost:8000/docs` using `/ingest` when your auth cookie is available.
12. Run your first `/analyze` request.
13. How to generate values

SECRET_KEY: python -c "import secrets; print(secrets.token_urlsafe(48))"
FERNET_KEY: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
