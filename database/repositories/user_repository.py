"""CRUD operations for the User entity (see src/services/auth_service.py
for password hashing/verification -- this repository never handles plain-
text passwords, only rows)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import User


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, **fields) -> User:
        user = User(**fields)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)

    def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        return self.session.scalars(stmt).first()

    def list(self) -> list[User]:
        stmt = select(User).order_by(User.created_at)
        return list(self.session.scalars(stmt).all())

    def count(self) -> int:
        return len(self.list())

    def update_last_login(self, user_id: int, when) -> None:
        user = self.get(user_id)
        if user is not None:
            user.last_login_at = when
            self.session.commit()

    def delete(self, user_id: int) -> bool:
        user = self.get(user_id)
        if user is None:
            return False
        self.session.delete(user)
        self.session.commit()
        return True
