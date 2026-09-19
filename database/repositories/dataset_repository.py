"""CRUD operations for the Dataset entity."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Dataset


class DatasetRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, **fields) -> Dataset:
        dataset = Dataset(**fields)
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def get(self, dataset_id: int) -> Dataset | None:
        return self.session.get(Dataset, dataset_id)

    def list(self) -> list[Dataset]:
        return list(self.session.scalars(select(Dataset)).all())

    def update_status(self, dataset_id: int, status: str) -> Dataset | None:
        dataset = self.get(dataset_id)
        if dataset is None:
            return None
        dataset.status = status
        self.session.commit()
        self.session.refresh(dataset)
        return dataset

    def delete(self, dataset_id: int) -> bool:
        dataset = self.get(dataset_id)
        if dataset is None:
            return False
        self.session.delete(dataset)
        self.session.commit()
        return True
