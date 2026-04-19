from datetime import datetime, timedelta, timezone
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from auth.schemas import Token, UserCreate, UserOut
from auth.utils import ALGORITHM, create_access_token, hash_password, verify_password
from config import settings
from database import get_db


router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def _google_sso_configured() -> bool:
    return bool(
        settings.GOOGLE_OAUTH_CLIENT_ID
        and settings.GOOGLE_OAUTH_CLIENT_SECRET
        and settings.GOOGLE_OAUTH_REDIRECT_URI
    )


def _create_google_state() -> str:
    payload = {
        "purpose": "google_oauth_state",
        "nonce": secrets.token_urlsafe(12),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def _validate_google_state(state: str) -> bool:
    try:
        payload = jwt.decode(state, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return False

    return payload.get("purpose") == "google_oauth_state"


def _frontend_login_redirect(error: str | None = None) -> str:
    base = f"{settings.FRONTEND_URL.rstrip('/')}/login"

    params: dict[str, str] = {}
    if error:
        params["error"] = error

    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def _frontend_post_login_redirect() -> str:
    return f"{settings.FRONTEND_URL.rstrip('/')}/analyze"


def _cookie_samesite() -> str:
    value = settings.AUTH_COOKIE_SAMESITE.strip().lower()
    if value in {"lax", "strict", "none"}:
        return value
    return "lax"


def _set_auth_cookie(response: Response, access_token: str) -> None:
    max_age = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=access_token,
        max_age=max_age,
        expires=max_age,
        path="/",
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite=_cookie_samesite(),
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path="/",
        secure=settings.AUTH_COOKIE_SECURE,
        samesite=_cookie_samesite(),
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)) -> UserOut:
    if settings.AUTH_MODE == "google_sso":
        raise HTTPException(status_code=405, detail="Password registration is disabled. Use Google SSO.")

    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=user_data.email, hashed_password=hash_password(user_data.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    if settings.AUTH_MODE == "google_sso":
        raise HTTPException(status_code=405, detail="Password login is disabled. Use Google SSO.")

    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)},
        expires_delta=access_token_expires,
    )

    _set_auth_cookie(response, access_token)

    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)


@router.post("/logout")
async def logout() -> JSONResponse:
    response = JSONResponse({"message": "logged_out"})
    _clear_auth_cookie(response)
    return response


@router.get("/google/login")
async def google_login() -> RedirectResponse:
    if settings.AUTH_MODE != "google_sso":
        raise HTTPException(status_code=404, detail="Google SSO endpoint is disabled")

    if not _google_sso_configured():
        raise HTTPException(status_code=503, detail="Google SSO is not configured on the server")

    state = _create_google_state()
    params = {
        "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": settings.GOOGLE_OAUTH_SCOPES,
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }

    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)


@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    if settings.AUTH_MODE != "google_sso":
        return RedirectResponse(_frontend_login_redirect(error="google_sso_disabled"), status_code=302)

    if error:
        return RedirectResponse(_frontend_login_redirect(error="google_oauth_denied"), status_code=302)

    if not _google_sso_configured():
        return RedirectResponse(_frontend_login_redirect(error="google_sso_not_configured"), status_code=302)

    if not code or not state or not _validate_google_state(state):
        return RedirectResponse(_frontend_login_redirect(error="invalid_google_oauth_state"), status_code=302)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
                    "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
        token_response.raise_for_status()

        token_payload = token_response.json()
        id_token = str(token_payload.get("id_token") or "").strip()
        if not id_token:
            raise ValueError("Missing id_token in Google token response")

        async with httpx.AsyncClient(timeout=10.0) as client:
            token_info_response = await client.get(GOOGLE_TOKENINFO_URL, params={"id_token": id_token})
        token_info_response.raise_for_status()

        token_info = token_info_response.json()
        audience = str(token_info.get("aud") or "").strip()
        email = str(token_info.get("email") or "").strip().lower()
        email_verified = str(token_info.get("email_verified") or "").strip().lower() in {"true", "1"}

        if audience != settings.GOOGLE_OAUTH_CLIENT_ID:
            raise ValueError("Google token audience mismatch")
        if not email or not email_verified:
            raise ValueError("Google account email is missing or unverified")
    except Exception:
        return RedirectResponse(_frontend_login_redirect(error="google_auth_failed"), status_code=302)

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            email=email,
            hashed_password=hash_password(f"sso::{secrets.token_urlsafe(24)}"),
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)},
        expires_delta=access_token_expires,
    )

    response = RedirectResponse(_frontend_post_login_redirect(), status_code=302)
    _set_auth_cookie(response, access_token)
    return response