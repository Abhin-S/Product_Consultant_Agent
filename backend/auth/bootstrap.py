from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from auth.models import User
from auth.utils import hash_password
from config import settings
from database import AsyncSessionLocal


logger = logging.getLogger(__name__)


async def ensure_default_admin() -> None:
    """
    Seed a default admin user when credentials are provided via env.
    This operation is idempotent and safe to run on every startup.
    """
    email = settings.DEFAULT_ADMIN_EMAIL.strip()
    password = settings.DEFAULT_ADMIN_PASSWORD

    if not email or not password:
        return

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.email == email))
            existing = result.scalar_one_or_none()

            if existing is None:
                user = User(
                    email=email,
                    hashed_password=hash_password(password),
                    is_admin=True,
                )
                db.add(user)
                await db.commit()
                logger.info("Seeded default admin user: %s", email)
                return

            if not existing.is_admin:
                existing.is_admin = True
                await db.commit()
                logger.info("Promoted existing user to admin: %s", email)
    except SQLAlchemyError as exc:
        logger.warning("Skipping default admin seed; database not ready yet: %s", exc)
