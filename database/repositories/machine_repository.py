"""CRUD operations for the Machine entity."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Machine


class MachineRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, **fields) -> Machine:
        machine = Machine(**fields)
        self.session.add(machine)
        self.session.commit()
        self.session.refresh(machine)
        return machine

    def get(self, machine_id: int) -> Machine | None:
        return self.session.get(Machine, machine_id)

    def get_by_identifier(self, machine_identifier: str) -> Machine | None:
        stmt = select(Machine).where(Machine.machine_identifier == machine_identifier)
        return self.session.scalars(stmt).first()

    def list(self, dataset_id: int | None = None) -> list[Machine]:
        stmt = select(Machine)
        if dataset_id is not None:
            stmt = stmt.where(Machine.dataset_id == dataset_id)
        return list(self.session.scalars(stmt).all())

    def delete(self, machine_id: int) -> bool:
        machine = self.get(machine_id)
        if machine is None:
            return False
        self.session.delete(machine)
        self.session.commit()
        return True
