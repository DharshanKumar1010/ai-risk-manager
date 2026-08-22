"""Declarative base shared by every RiskIQ ORM model."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class that every RiskIQ ORM model inherits from.

    Table modules under ``app/models/`` register themselves against this metadata, so
    importing them is what makes them visible to Alembic autogeneration.
    """
