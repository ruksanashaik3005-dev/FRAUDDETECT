"""SQLAlchemy persistence models for evaluated transactions and alerts."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin', 'analyst')", name="valid_user_role"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="analyst")
    token_version: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[float] = mapped_column(Float)
    payment_method: Mapped[str] = mapped_column(String(40), default="Unknown")
    account_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    device_information: Mapped[str] = mapped_column(String(255), default="Unknown")
    amount_to_user_avg_ratio: Mapped[float] = mapped_column(Float, default=1.0)
    is_high_risk_country_and_vpn: Mapped[int] = mapped_column(Integer, default=0)
    velocity_amount_product: Mapped[float] = mapped_column(Float, default=0.0)
    hour: Mapped[int] = mapped_column(Integer, default=0)
    day_of_week: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    location: Mapped[str] = mapped_column(String(100))
    merchant_category: Mapped[str] = mapped_column(String(80))
    is_vpn: Mapped[bool] = mapped_column(Boolean, default=False)
    transaction_frequency_1h: Mapped[int] = mapped_column(Integer, default=0)
    fraud_score: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_level: Mapped[str] = mapped_column(String(16), default="Low")
    actual_is_fraud: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    prediction: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(32), default="Approved")
    reason_codes: Mapped[str] = mapped_column(Text, default="[]")

    alert: Mapped[Alert | None] = relationship(
        back_populates="transaction", cascade="all, delete-orphan", uselist=False
    )
    fraud_prediction: Mapped[FraudPrediction | None] = relationship(
        back_populates="transaction", cascade="all, delete-orphan", uselist=False
    )


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.transaction_id"), unique=True, index=True
    )
    priority: Mapped[str] = mapped_column(String(16), default="High")
    status: Mapped[str] = mapped_column(String(24), default="Open")
    severity: Mapped[str] = mapped_column(String(16), default="High")
    detection_reason: Mapped[str] = mapped_column(Text, default="[]")
    investigation_notes: Mapped[str] = mapped_column(Text, default="")
    assigned_analyst: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    transaction: Mapped[Transaction] = relationship(back_populates="alert")


class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.transaction_id"), unique=True, index=True
    )
    prediction: Mapped[str] = mapped_column(String(24))
    fraud_score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    reason_codes: Mapped[str] = mapped_column(Text, default="[]")
    model_version: Mapped[str] = mapped_column(String(80), default="synthetic-demo")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    transaction: Mapped[Transaction] = relationship(back_populates="fraud_prediction")


class UserActivity(Base):
    __tablename__ = "user_activities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str] = mapped_column(String(255))
    details: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class SecurityLog(Base):
    __tablename__ = "security_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="Info")
    summary: Mapped[str] = mapped_column(String(255))
    details: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('Low', 'Medium', 'High', 'Critical')",
            name="valid_incident_severity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    incident_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16), default="Medium")
    assigned_analyst: Mapped[str | None] = mapped_column(String(64), nullable=True)
    investigation_notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="Triage")
    resolution_details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class InvestigationRecord(Base):
    __tablename__ = "investigation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str | None] = mapped_column(
        ForeignKey("transactions.transaction_id"), index=True, nullable=True
    )
    alert_id: Mapped[int | None] = mapped_column(
        ForeignKey("alerts.id"), index=True, nullable=True
    )
    incident_id: Mapped[str | None] = mapped_column(
        ForeignKey("incidents.incident_id"), index=True, nullable=True
    )
    analyst: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(40))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class TestResult(Base):
    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(String(40), index=True)
    description: Mapped[str] = mapped_column(String(255))
    input_data: Mapped[str] = mapped_column(Text, default="{}")
    expected_result: Mapped[str] = mapped_column(Text)
    actual_result: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
