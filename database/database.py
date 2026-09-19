"""SQLAlchemy engine/session setup for the SQLite thesis-prototype database.

Only this module knows the connection string; everything else (repositories,
services) imports `SessionLocal` / `get_session` from here so the database
backend can be swapped (e.g. to PostgreSQL) without touching calling code.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import get_database_url


class Base(DeclarativeBase):
    pass


_connect_args = {}
if get_database_url().startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(get_database_url(), connect_args=_connect_args)
# expire_on_commit=False: callers (esp. Streamlit pages) routinely read
# attributes off objects fetched inside a `with get_session()` block after
# the block (and its commit) has already happened. With the default
# expire_on_commit=True, that access re-triggers a lazy load against the
# now-closed session and raises DetachedInstanceError.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables that don't already exist."""
    from database import models  # noqa: F401  (ensure models are registered)

    Base.metadata.create_all(bind=engine)
