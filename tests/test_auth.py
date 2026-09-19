"""Tests for src/services/auth_service.py: password hashing/verification
and account creation/login, against an isolated in-memory database."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base
from src.services.auth_service import (
    MIN_PASSWORD_LENGTH,
    any_users_exist,
    authenticate,
    create_user,
    hash_password,
    validate_password,
    validate_username,
    verify_password,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    s = TestSession()
    try:
        yield s
    finally:
        s.close()


def test_hash_password_never_stores_plaintext():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")


def test_verify_password_true_for_correct_password():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed) is True


def test_verify_password_false_for_wrong_password():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("wrong password", hashed) is False


def test_verify_password_false_for_corrupted_hash_never_raises():
    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


def test_validate_username_rejects_too_short():
    assert validate_username("ab") is not None


def test_validate_username_accepts_reasonable_name():
    assert validate_username("ama.lebbie") is None


def test_validate_password_enforces_minimum_length():
    assert validate_password("short") is not None
    assert validate_password("a" * MIN_PASSWORD_LENGTH) is None


def test_any_users_exist_false_on_empty_database(session):
    assert any_users_exist(session) is False


def test_create_user_then_any_users_exist_true(session):
    result = create_user(session, "ama", "Ama Lebbie", "a-strong-password")
    assert result.ok is True
    assert any_users_exist(session) is True


def test_create_user_rejects_duplicate_username(session):
    create_user(session, "ama", "Ama Lebbie", "a-strong-password")
    result = create_user(session, "ama", "Someone Else", "another-password")
    assert result.ok is False
    assert "already taken" in result.error


def test_create_user_rejects_weak_password(session):
    result = create_user(session, "ama", "Ama Lebbie", "short")
    assert result.ok is False


def test_authenticate_succeeds_with_correct_credentials(session):
    create_user(session, "ama", "Ama Lebbie", "a-strong-password")
    result = authenticate(session, "ama", "a-strong-password")
    assert result.ok is True
    assert result.user.username == "ama"
    assert result.user.last_login_at is not None


def test_authenticate_fails_with_wrong_password(session):
    create_user(session, "ama", "Ama Lebbie", "a-strong-password")
    result = authenticate(session, "ama", "wrong-password")
    assert result.ok is False


def test_authenticate_fails_for_unknown_username(session):
    result = authenticate(session, "nobody", "whatever-password")
    assert result.ok is False


def test_authenticate_error_message_does_not_reveal_which_field_was_wrong(session):
    create_user(session, "ama", "Ama Lebbie", "a-strong-password")
    wrong_password = authenticate(session, "ama", "wrong-password")
    unknown_user = authenticate(session, "nobody", "whatever-password")
    assert wrong_password.error == unknown_user.error
