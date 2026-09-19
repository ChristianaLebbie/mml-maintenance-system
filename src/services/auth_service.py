"""Authentication: named-account login for the app (see database.models.User
and database.repositories.user_repository.UserRepository).

Kept intentionally small -- this protects a small team's shared
decision-support tool, not a multi-tenant public product: every signed-in
user has the same access, `role` is a display label rather than an
enforced permission, and there is no password-reset-by-email flow (a
locked-out user gets a new one made for them by someone already signed
in). Passwords are hashed with bcrypt and a plain-text password is never
stored, logged, or returned by any function here.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass

import bcrypt
from sqlalchemy.orm import Session

from database.models import User
from database.repositories.user_repository import UserRepository

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,50}$")
MIN_PASSWORD_LENGTH = 8


@dataclass
class AuthResult:
    ok: bool
    user: User | None = None
    error: str | None = None


def validate_username(username: str) -> str | None:
    if not USERNAME_PATTERN.match(username or ""):
        return "Username must be 3-50 characters: letters, numbers, dots, underscores, or hyphens only."
    return None


def validate_password(password: str) -> str | None:
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed/corrupted hash must never be treated as a match.
        return False


def any_users_exist(session: Session) -> bool:
    return UserRepository(session).count() > 0


def create_user(
    session: Session, username: str, display_name: str, password: str, role: str = "Member"
) -> AuthResult:
    username = (username or "").strip()
    display_name = (display_name or "").strip() or username

    problem = validate_username(username) or validate_password(password)
    if problem:
        return AuthResult(ok=False, error=problem)

    repo = UserRepository(session)
    if repo.get_by_username(username) is not None:
        return AuthResult(ok=False, error=f"Username '{username}' is already taken.")

    user = repo.create(
        username=username,
        display_name=display_name,
        password_hash=hash_password(password),
        role=role,
    )
    return AuthResult(ok=True, user=user)


def authenticate(session: Session, username: str, password: str) -> AuthResult:
    repo = UserRepository(session)
    user = repo.get_by_username((username or "").strip())
    # Deliberately the same generic error for "no such user" and "wrong
    # password" -- confirming which one it was would let an attacker
    # enumerate valid usernames.
    if user is None or not verify_password(password or "", user.password_hash):
        return AuthResult(ok=False, error="Incorrect username or password.")

    repo.update_last_login(user.id, datetime.datetime.utcnow())
    return AuthResult(ok=True, user=user)
