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

Fill the required values in `.env` (at least `SECRET_KEY`, `GOOGLE_API_KEY`, `FERNET_KEY`).

Set default admin credentials in `backend/.env` to auto-seed an admin on startup:

```dotenv
DEFAULT_ADMIN_EMAIL=admin@example.com
DEFAULT_ADMIN_PASSWORD=replace_with_strong_password
```

Start backend:

```powershell
uvicorn main:app --reload --port 8000
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

- Chroma documents/chunks are created only after you run the backend and call the `/ingest` endpoint.
- The root `.venv/` is not required for this setup path. The recommended Python environment is `backend/.venv`.

## First Run Checklist

1. Create and activate backend venv in `backend/`.
2. Install backend dependencies from `backend/requirements.txt`.
3. Copy `backend/.env.example` to `backend/.env` and fill required keys.
4. Set `DEFAULT_ADMIN_EMAIL` and `DEFAULT_ADMIN_PASSWORD` in `backend/.env`.
5. Run Alembic migrations from `backend/`:

```powershell
alembic upgrade head
```

6. Start backend:

```powershell
uvicorn main:app --reload --port 8000
```

7. In `frontend/`, run `npm install`, copy `.env.local.example` to `.env.local`, then run `npm run dev`.
8. Log in as default admin and call `/ingest` once to build the local vector index.
9. Run your first `/analyze` request.
10. How to generate values

SECRET_KEY: python -c "import secrets; print(secrets.token_urlsafe(48))"
FERNET_KEY: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
