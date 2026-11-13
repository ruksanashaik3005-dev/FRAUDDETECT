"""Authenticated FastAPI service for transaction scoring and alert review."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Response,
    Security,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sqlalchemy import URL, and_, create_engine, desc, func, inspect, or_, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from ml_pipeline import MODEL_PATH, prepare_features, train_model
from models import (
    Alert,
    Base,
    FraudPrediction,
    Incident,
    InvestigationRecord,
    SecurityLog,
    TestResult,
    Transaction,
    User,
    UserActivity,
)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
else:
    DATABASE_URL = URL.create(
        "postgresql+psycopg",
        username=os.getenv("POSTGRES_USER", "frauddetect"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "frauddect"),
    )
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
logger = logging.getLogger("frauddetect.startup")

JWT_SECRET_KEY = os.getenv("FRAUDDETECT_JWT_SECRET", "")
API_KEY = os.getenv("FRAUDDETECT_API_KEY", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("FRAUDDETECT_TOKEN_MINUTES", "30"))
ADMIN_USERNAME = os.getenv("FRAUDDETECT_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("FRAUDDETECT_ADMIN_PASSWORD", "")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TransactionInput(BaseModel):
    transaction_id: str = Field(default_factory=lambda: f"TX-{uuid4().hex[:12].upper()}")
    user_id: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=10_000_000)
    payment_method: str = Field(default="Unknown", min_length=1, max_length=40)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    account_balance: float | None = Field(default=None, ge=0, le=100_000_000)
    location: str = Field(min_length=1, max_length=100)
    device_information: str = Field(default="Unknown", max_length=255)
    merchant_category: str = Field(min_length=1, max_length=80)
    is_vpn: bool = False
    transaction_frequency_1h: int = Field(default=0, ge=0, le=10_000)


class RegistrationInput(BaseModel):
    username: str = Field(
        min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    password: str = Field(min_length=12, max_length=128)

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return value


class ActionInput(BaseModel):
    action: Literal["Blocked", "Reviewed/Genuine"]


class AlertInvestigationInput(BaseModel):
    status: Literal["Investigating", "Resolved"]
    notes: str = Field(min_length=1, max_length=4000)


class TransactionOutcomeInput(BaseModel):
    actual_is_fraud: bool
    notes: str = Field(min_length=1, max_length=4000)


class IncidentInput(BaseModel):
    description: str = Field(min_length=8, max_length=4000)
    severity: Literal["Low", "Medium", "High", "Critical"] = "Medium"
    assigned_analyst: str | None = Field(default=None, max_length=64)


class IncidentUpdateInput(BaseModel):
    description: str | None = Field(default=None, min_length=8, max_length=4000)
    severity: Literal["Low", "Medium", "High", "Critical"] | None = None
    assigned_analyst: str | None = Field(default=None, max_length=64)
    investigation_notes: str | None = Field(default=None, max_length=8000)
    status: Literal[
        "Triage", "Contained", "Evidence Preserved", "Recovery", "Resolved"
    ] | None = None
    resolution_details: str | None = Field(default=None, max_length=4000)


class UserAdminUpdateInput(BaseModel):
    role: Literal["admin", "analyst"] | None = None
    is_active: bool | None = None


class TransactionOutput(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: str
    user_id: str
    amount: float
    payment_method: str
    timestamp: datetime
    account_balance: float | None
    location: str
    device_information: str
    merchant_category: str
    is_vpn: bool
    transaction_frequency_1h: int
    fraud_score: float
    confidence_score: float
    anomaly_score: float
    is_anomaly: bool
    risk_level: str
    prediction: str
    status: str
    reason_codes: list[str]
    suspicious_features: list[str]
    detection_explanation: str


class TokenOutput(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Literal["admin", "analyst"]
    expires_in: int


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, event: dict) -> None:
        for connection in tuple(self.connections):
            try:
                await connection.send_json(event)
            except (OSError, RuntimeError, WebSocketDisconnect):
                self.disconnect(connection)


manager = ConnectionManager()
model_bundle: dict = {}


def _validate_startup_configuration() -> None:
    missing = [
        name for name, value in (
            ("FRAUDDETECT_JWT_SECRET", JWT_SECRET_KEY),
            ("FRAUDDETECT_API_KEY", API_KEY),
            ("FRAUDDETECT_ADMIN_PASSWORD", ADMIN_PASSWORD),
        ) if not value
    ]
    if missing:
        raise RuntimeError(
            "Set required security environment variables before startup: "
            + ", ".join(missing)
        )
    if len(JWT_SECRET_KEY) < 32:
        raise RuntimeError("FRAUDDETECT_JWT_SECRET must contain at least 32 characters")
    if len(API_KEY) < 24:
        raise RuntimeError("FRAUDDETECT_API_KEY must contain at least 24 characters")
    if len(ADMIN_PASSWORD) < 12:
        raise RuntimeError("FRAUDDETECT_ADMIN_PASSWORD must contain at least 12 characters")
    if len(ADMIN_PASSWORD.encode("utf-8")) > 72:
        raise RuntimeError("FRAUDDETECT_ADMIN_PASSWORD must fit bcrypt's 72-byte input limit")
    if ACCESS_TOKEN_MINUTES < 1:
        raise RuntimeError("FRAUDDETECT_TOKEN_MINUTES must be a positive number")
    if not os.getenv("DATABASE_URL") and not os.getenv("POSTGRES_PASSWORD"):
        raise RuntimeError("Set DATABASE_URL or POSTGRES_PASSWORD before startup")


def _wait_for_database() -> None:
    max_attempts = int(os.getenv("DB_STARTUP_MAX_ATTEMPTS", "30"))
    base_delay = float(os.getenv("DB_STARTUP_RETRY_SECONDS", "1"))
    if max_attempts < 1 or base_delay < 0:
        raise RuntimeError("Database retry attempts must be positive and delay non-negative")
    for attempt in range(1, max_attempts + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Database connection established")
            return
        except OperationalError as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Database was unavailable after {max_attempts} startup attempts"
                ) from error
            delay = min(base_delay * (2 ** (attempt - 1)), 10.0)
            logger.warning(
                "Database not ready (attempt %s/%s); retrying in %.1f seconds",
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)


def _migrate_inference_columns() -> None:
    if engine.dialect.name not in {"postgresql", "sqlite"}:
        return
    definitions = {
        "users": {
            "token_version": "INTEGER NOT NULL DEFAULT 0",
            "is_active": "BOOLEAN NOT NULL DEFAULT TRUE",
        },
        "transactions": {
            "payment_method": "VARCHAR(40) NOT NULL DEFAULT 'Unknown'",
            "account_balance": "FLOAT",
            "device_information": "VARCHAR(255) NOT NULL DEFAULT 'Unknown'",
            "amount_to_user_avg_ratio": "FLOAT NOT NULL DEFAULT 1.0",
            "is_high_risk_country_and_vpn": "INTEGER NOT NULL DEFAULT 0",
            "velocity_amount_product": "FLOAT NOT NULL DEFAULT 0.0",
            "hour": "INTEGER NOT NULL DEFAULT 0",
            "day_of_week": "INTEGER NOT NULL DEFAULT 0",
            "confidence_score": "FLOAT NOT NULL DEFAULT 0.0",
            "anomaly_score": "FLOAT NOT NULL DEFAULT 0.0",
            "is_anomaly": "BOOLEAN NOT NULL DEFAULT FALSE",
            "risk_level": "VARCHAR(16) NOT NULL DEFAULT 'Low'",
            "actual_is_fraud": "BOOLEAN",
        },
        "alerts": {
            "severity": "VARCHAR(16) NOT NULL DEFAULT 'High'",
            "detection_reason": "TEXT NOT NULL DEFAULT '[]'",
            "investigation_notes": "TEXT NOT NULL DEFAULT ''",
            "assigned_analyst": "VARCHAR(64)",
            "resolved_at": "TIMESTAMP",
        },
        "investigation_records": {
            "transaction_id": "VARCHAR(64) REFERENCES transactions(transaction_id)",
        },
    }
    previously_existing: dict[str, set[str]] = {}
    with engine.begin() as connection:
        for table_name, columns in definitions.items():
            existing = {
                column["name"] for column in inspect(connection).get_columns(table_name)
            }
            previously_existing[table_name] = existing
            for column_name, definition in columns.items():
                if column_name in existing:
                    continue
                clause = " IF NOT EXISTS" if engine.dialect.name == "postgresql" else ""
                connection.execute(
                    text(
                        f"ALTER TABLE {table_name} ADD COLUMN{clause} "
                        f"{column_name} {definition}"
                    )
                )
        existing_columns = previously_existing["transactions"]
        if "amount_to_user_avg_ratio" not in existing_columns:
            greatest = "GREATEST" if engine.dialect.name == "postgresql" else "MAX"
            connection.execute(text(
                "UPDATE transactions AS current SET amount_to_user_avg_ratio = "
                f"current.amount / {greatest}(COALESCE(("
                "SELECT AVG(previous.amount) FROM transactions AS previous "
                "WHERE previous.user_id = current.user_id "
                "AND previous.id <> current.id), current.amount), 1.0)"
            ))
        if "is_high_risk_country_and_vpn" not in existing_columns:
            connection.execute(text(
                "UPDATE transactions SET is_high_risk_country_and_vpn = "
                "CASE WHEN is_vpn AND location IN ('Lagos', 'Moscow', 'Tehran', 'Pyongyang') "
                "THEN 1 ELSE 0 END"
            ))
        if "velocity_amount_product" not in existing_columns:
            connection.execute(text(
                "UPDATE transactions SET velocity_amount_product = "
                "amount * transaction_frequency_1h"
            ))
        if "hour" not in existing_columns:
            hour_expression = (
                "EXTRACT(HOUR FROM timestamp)::INTEGER"
                if engine.dialect.name == "postgresql"
                else "CAST(strftime('%H', timestamp) AS INTEGER)"
            )
            connection.execute(text(f"UPDATE transactions SET hour = {hour_expression}"))
        if "day_of_week" not in existing_columns:
            weekday_expression = (
                "EXTRACT(ISODOW FROM timestamp)::INTEGER - 1"
                if engine.dialect.name == "postgresql"
                else "((CAST(strftime('%w', timestamp) AS INTEGER) + 6) % 7)"
            )
            connection.execute(
                text(f"UPDATE transactions SET day_of_week = {weekday_expression}")
            )
        if "confidence_score" not in existing_columns:
            connection.execute(text(
                "UPDATE transactions SET confidence_score = "
                "CASE WHEN fraud_score >= 0.5 THEN fraud_score ELSE 1.0 - fraud_score END"
            ))
        if "risk_level" not in existing_columns:
            connection.execute(text(
                "UPDATE transactions SET risk_level = CASE "
                "WHEN fraud_score >= 0.95 THEN 'Critical' "
                "WHEN fraud_score >= 0.75 THEN 'High' "
                "WHEN fraud_score >= 0.4 THEN 'Medium' ELSE 'Low' END"
            ))
        if "detection_reason" not in previously_existing["alerts"]:
            connection.execute(text(
                "UPDATE alerts SET detection_reason = COALESCE(("
                "SELECT transactions.reason_codes FROM transactions "
                "WHERE transactions.transaction_id = alerts.transaction_id), '[]')"
            ))
        if "severity" not in previously_existing["alerts"]:
            connection.execute(text(
                "UPDATE alerts SET severity = COALESCE(("
                "SELECT CASE WHEN transactions.fraud_score >= 0.95 THEN 'Critical' "
                "WHEN transactions.fraud_score >= 0.75 THEN 'High' "
                "WHEN transactions.fraud_score >= 0.4 THEN 'Medium' ELSE 'Low' END "
                "FROM transactions WHERE transactions.transaction_id = alerts.transaction_id), "
                "'High')"
            ))
        if "transaction_id" not in previously_existing["investigation_records"]:
            connection.execute(text(
                "UPDATE investigation_records SET transaction_id = COALESCE(("
                "SELECT alerts.transaction_id FROM alerts "
                "WHERE alerts.id = investigation_records.alert_id), transaction_id)"
            ))


def _seed_admin() -> None:
    with SessionLocal() as db:
        if db.scalar(select(User.id).where(User.username == ADMIN_USERNAME)) is None:
            db.add(
                User(
                    username=ADMIN_USERNAME,
                    hashed_password=pwd_context.hash(ADMIN_PASSWORD),
                    role="admin",
                )
            )
            db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model_bundle
    _validate_startup_configuration()
    _wait_for_database()
    Base.metadata.create_all(bind=engine)
    _migrate_inference_columns()
    _seed_admin()
    if MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and {
            "model",
            "threshold",
            "anomaly_preprocessor",
            "anomaly_detector",
        } <= loaded.keys():
            model_bundle = loaded
        else:
            train_model()
            model_bundle = joblib.load(MODEL_PATH)
    else:
        train_model()
        model_bundle = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="FraudDetect API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(","),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


def get_db():
    with SessionLocal() as db:
        yield db


def _decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    payload = _decode_token(token)
    user = db.scalar(select(User).where(User.username == payload["sub"]))
    if (
        user is None
        or not user.is_active
        or payload.get("ver") != user.token_version
    ):
        raise HTTPException(
            status_code=401,
            detail="Session has expired or been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required")
    return user


def require_api_key(
    api_key: Annotated[str | None, Security(api_key_header)],
) -> None:
    if api_key is None or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _record_activity(
    db: Session,
    username: str | None,
    event_type: str,
    description: str,
    *,
    severity: str = "Info",
    details: dict | None = None,
) -> None:
    safe_details = json.dumps(details or {}, default=str)
    db.add(
        UserActivity(
            username=username,
            event_type=event_type,
            description=description,
            details=safe_details,
        )
    )
    db.add(
        SecurityLog(
            username=username,
            event_type=event_type,
            severity=severity,
            summary=description,
            details=safe_details,
        )
    )


def _risk_level(score: float, threshold: float) -> str:
    if score >= max(0.95, threshold):
        return "Critical"
    if score >= threshold:
        return "High"
    if score >= min(0.4, threshold):
        return "Medium"
    return "Low"


def _reason_codes(
    transaction: TransactionInput,
    *,
    amount_ratio: float,
    is_anomaly: bool,
) -> list[str]:
    reasons = []
    if transaction.transaction_frequency_1h >= 5:
        reasons.append("High Frequency")
    if transaction.is_vpn:
        reasons.append("VPN Detected")
    if transaction.location in {"Lagos", "Moscow", "Tehran", "Pyongyang"}:
        reasons.append("High-Risk Country")
    if transaction.amount >= 1_000:
        reasons.append("High Amount")
    if amount_ratio >= 6:
        reasons.append("Unusual Amount for User")
    if transaction.account_balance is not None and transaction.amount > transaction.account_balance:
        reasons.append("Amount Exceeds Supplied Balance")
    if is_anomaly:
        reasons.append("Anomaly Detector Signal")
    return reasons


def _serialize(row: Transaction) -> dict:
    reason_codes = json.loads(row.reason_codes)
    feature_names = {
        "High Frequency": "transaction_frequency_1h",
        "VPN Detected": "is_vpn",
        "High-Risk Country": "location",
        "High Amount": "amount",
        "Unusual Amount for User": "amount_to_user_avg_ratio",
        "Amount Exceeds Supplied Balance": "account_balance",
        "Anomaly Detector Signal": "is_anomaly",
    }
    explanation = (
        "; ".join(reason_codes)
        if reason_codes
        else "No configured rule or anomaly signal was triggered."
    )
    explanation += (
        " Payment method, account balance, and device information are saved for "
        "review; only trained model features contribute to the Random Forest score."
    )
    return {
        "transaction_id": row.transaction_id,
        "user_id": row.user_id,
        "amount": row.amount,
        "payment_method": row.payment_method,
        "timestamp": row.timestamp.isoformat(),
        "account_balance": row.account_balance,
        "location": row.location,
        "device_information": row.device_information,
        "merchant_category": row.merchant_category,
        "is_vpn": row.is_vpn,
        "transaction_frequency_1h": row.transaction_frequency_1h,
        "fraud_score": row.fraud_score,
        "confidence_score": row.confidence_score,
        "anomaly_score": row.anomaly_score,
        "is_anomaly": row.is_anomaly,
        "risk_level": row.risk_level,
        "prediction": row.prediction,
        "status": row.status,
        "reason_codes": reason_codes,
        "suspicious_features": [feature_names[reason] for reason in reason_codes],
        "detection_explanation": explanation,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login", response_model=TokenOutput)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> TokenOutput:
    user = db.scalar(select(User).where(User.username == form_data.username))
    if (
        user is None
        or not user.is_active
        or not pwd_context.verify(form_data.password, user.hashed_password)
    ):
        _record_activity(
            db,
            form_data.username[:64] or None,
            "login_failed",
            "Unsuccessful sign-in attempt",
            severity="Warning",
        )
        db.commit()
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    token = jwt.encode(
        {
            "sub": user.username,
            "role": user.role,
            "ver": user.token_version,
            "exp": expires,
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )
    _record_activity(db, user.username, "login", "User signed in")
    db.commit()
    return TokenOutput(
        access_token=token,
        role=user.role,
        expires_in=ACCESS_TOKEN_MINUTES * 60,
    )


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(
    payload: RegistrationInput,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    if db.scalar(select(User.id).where(User.username == payload.username)):
        raise HTTPException(status_code=409, detail="Username is already registered")
    user = User(
        username=payload.username,
        hashed_password=pwd_context.hash(payload.password),
        role="analyst",
    )
    db.add(user)
    _record_activity(
        db,
        payload.username,
        "registration",
        "Analyst account registered",
    )
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username is already registered") from error
    return {"username": user.username, "role": user.role}


@app.post("/api/auth/logout")
def logout(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    user.token_version += 1
    _record_activity(db, user.username, "logout", "User signed out; sessions revoked")
    db.commit()
    return {"status": "signed_out"}


async def _evaluate_transaction(
    payload: TransactionInput,
    db: Session,
    actor_username: str | None,
) -> TransactionOutput:
    normalized_time = payload.timestamp
    if normalized_time.tzinfo is None:
        normalized_time = normalized_time.replace(tzinfo=timezone.utc)
    normalized_time = normalized_time.astimezone(timezone.utc)
    payload = payload.model_copy(update={"timestamp": normalized_time})

    if db.scalar(select(Transaction.id).where(Transaction.transaction_id == payload.transaction_id)):
        raise HTTPException(status_code=409, detail="transaction_id already exists")
    user_average = db.scalar(
        select(func.avg(Transaction.amount)).where(Transaction.user_id == payload.user_id)
    )
    features = prepare_features(
        pd.DataFrame([payload.model_dump()]),
        {payload.user_id: user_average} if user_average is not None else {},
    )
    score = float(model_bundle["model"].predict_proba(features)[0, 1])
    threshold = float(model_bundle["threshold"])
    flagged = score >= threshold
    engineered = features.iloc[0]
    anomaly_features = model_bundle["anomaly_preprocessor"].transform(features)
    anomaly_detector = model_bundle["anomaly_detector"]
    anomaly_score = max(
        0.0,
        min(1.0, 0.5 - float(anomaly_detector.decision_function(anomaly_features)[0])),
    )
    is_anomaly = bool(anomaly_detector.predict(anomaly_features)[0] == -1)
    confidence_score = max(score, 1.0 - score)
    risk_level = _risk_level(score, threshold)
    reasons = _reason_codes(
        payload,
        amount_ratio=float(engineered["amount_to_user_avg_ratio"]),
        is_anomaly=is_anomaly,
    )
    row = Transaction(
        **payload.model_dump(),
        amount_to_user_avg_ratio=float(engineered["amount_to_user_avg_ratio"]),
        is_high_risk_country_and_vpn=int(engineered["is_high_risk_country_and_vpn"]),
        velocity_amount_product=float(engineered["velocity_amount_product"]),
        hour=int(engineered["hour"]),
        day_of_week=int(engineered["day_of_week"]),
        fraud_score=score,
        confidence_score=confidence_score,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        risk_level=risk_level,
        prediction="Fraudulent" if flagged else "Genuine",
        status="Flagged-Fraudulent" if flagged else "Approved",
        reason_codes=json.dumps(reasons),
    )
    db.add(row)
    db.add(
        FraudPrediction(
            transaction_id=row.transaction_id,
            prediction=row.prediction,
            fraud_score=score,
            threshold=threshold,
            confidence_score=confidence_score,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            reason_codes=json.dumps(reasons),
            model_version=str(model_bundle.get("model_version", "synthetic-demo")),
        )
    )
    if flagged:
        row.alert = Alert(
            transaction_id=row.transaction_id,
            severity=risk_level,
            detection_reason=json.dumps(reasons),
        )
    _record_activity(
        db,
        actor_username,
        "transaction_analysis",
        "Transaction evaluated",
        severity="Warning" if flagged else "Info",
        details={"transaction_id": row.transaction_id, "risk_level": risk_level},
    )
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail="transaction_id already exists") from error
    db.refresh(row)
    result = TransactionOutput(**_serialize(row))

    await manager.broadcast({"type": "transaction", "transaction": result.model_dump(mode="json")})
    return result


@app.post(
    "/api/transactions/evaluate",
    response_model=TransactionOutput,
    dependencies=[Depends(require_api_key)],
)
async def evaluate_transaction(
    payload: TransactionInput,
    db: Annotated[Session, Depends(get_db)],
) -> TransactionOutput:
    return await _evaluate_transaction(payload, db, None)


@app.post("/api/transactions/analyze", response_model=TransactionOutput)
async def analyze_transaction(
    payload: TransactionInput,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> TransactionOutput:
    return await _evaluate_transaction(payload, db, user.username)


@app.get("/api/dashboard/stats")
def dashboard_stats(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=24)
    rows = db.scalars(select(Transaction).where(Transaction.timestamp >= since)).all()
    threshold = float(model_bundle["threshold"])
    flagged = [row for row in rows if row.fraud_score >= threshold]
    total = len(rows)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    trend_start = current_hour - timedelta(hours=23)
    trend_counts = [0] * 24
    category_counts: dict[str, int] = {}
    for row in rows:
        timestamp = row.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        if row.fraud_score >= threshold:
            hour_index = int(
                (timestamp.replace(minute=0, second=0, microsecond=0) - trend_start)
                .total_seconds()
                // 3600
            )
            if 0 <= hour_index < 24:
                trend_counts[hour_index] += 1
        category_counts[row.merchant_category] = category_counts.get(row.merchant_category, 0) + 1
    return {
        "total_transactions": total,
        "genuine_transaction_count": total - len(flagged),
        "fraudulent_flagged_count": len(flagged),
        "fraud_percentage": round(100 * len(flagged) / total, 2) if total else 0.0,
        "total_value_flagged": round(sum(row.amount for row in flagged), 2),
        "model_accuracy": round(float(model_bundle.get("metrics", {}).get("accuracy", 0)) * 100, 2),
        "fraud_threshold": threshold,
        "model_metrics": {
            metric: float(model_bundle.get("metrics", {}).get(metric, 0))
            for metric in (
                "precision", "recall", "f1_score", "roc_auc",
                "validation_precision", "validation_recall",
                "false_positive_rate", "false_negative_rate",
            )
        },
        "model_evaluation_source": model_bundle.get("metrics", {}).get(
            "evaluation_dataset", "synthetic_demo"
        ),
        "confusion_matrix": model_bundle.get("metrics", {}).get("confusion_matrix"),
        "hourly_fraud_trend": [
            {"hour": (trend_start + timedelta(hours=index)).strftime("%H:00"), "flagged": count}
            for index, count in enumerate(trend_counts)
        ],
        "category_counts": [
            {"name": name, "count": count}
            for name, count in sorted(
                category_counts.items(), key=lambda entry: entry[1], reverse=True
            )
        ],
    }


@app.get("/api/dashboard/analytics")
def dashboard_analytics(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    days: int = Query(default=30, ge=1, le=365),
) -> dict:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    rows = db.scalars(
        select(Transaction).where(Transaction.timestamp >= start)
    ).all()
    threshold = float(model_bundle["threshold"])
    fraud_rows = [row for row in rows if row.fraud_score >= threshold]
    genuine_rows = len(rows) - len(fraud_rows)
    daily_counts: dict[str, dict[str, int]] = {}
    weekly_counts: dict[str, int] = {}
    monthly_counts: dict[str, int] = {}
    amount_ranges = {
        "≤ $50": 0,
        "$50–$250": 0,
        "$250–$1,000": 0,
        "> $1,000": 0,
    }
    for row in rows:
        timestamp = row.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        day = timestamp.date().isoformat()
        if day not in daily_counts:
            daily_counts[day] = {"transactions": 0, "fraudulent": 0}
        daily_counts[day]["transactions"] += 1
        if row.fraud_score >= threshold:
            daily_counts[day]["fraudulent"] += 1
            week_key = (timestamp - timedelta(days=timestamp.weekday())).date().isoformat()
            weekly_counts[week_key] = weekly_counts.get(week_key, 0) + 1
            month_key = timestamp.strftime("%Y-%m")
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        if row.amount <= 50:
            amount_ranges["≤ $50"] += 1
        elif row.amount <= 250:
            amount_ranges["$50–$250"] += 1
        elif row.amount <= 1_000:
            amount_ranges["$250–$1,000"] += 1
        else:
            amount_ranges["> $1,000"] += 1

    alerts = db.scalars(
        select(Alert).where(Alert.created_at >= start)
    ).all()
    severity_counts = {level: 0 for level in ("Low", "Medium", "High", "Critical")}
    for alert in alerts:
        if alert.severity in severity_counts:
            severity_counts[alert.severity] += 1
    labeled_rows = [row for row in rows if row.actual_is_fraud is not None]
    production_metrics = None
    if labeled_rows:
        y_true = [int(row.actual_is_fraud) for row in labeled_rows]
        y_pred = [int(row.fraud_score >= threshold) for row in labeled_rows]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        production_metrics = {
            "labeled_samples": len(labeled_rows),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        }
    return {
        "window_days": days,
        "generated_at": now.isoformat(),
        "genuine_vs_fraudulent": [
            {"name": "Genuine", "value": genuine_rows},
            {"name": "Fraudulent", "value": len(fraud_rows)},
        ],
        "daily_activity": [
            {"date": day, **daily_counts[day]}
            for day in sorted(daily_counts)
        ],
        "weekly_fraud_trends": [
            {"week_start": week, "fraudulent": count}
            for week, count in sorted(weekly_counts.items())
        ],
        "monthly_fraud_statistics": [
            {"month": month, "fraudulent": count}
            for month, count in sorted(monthly_counts.items())
        ],
        "amount_distribution": [
            {"range": label, "count": count}
            for label, count in amount_ranges.items()
        ],
        "severity_distribution": [
            {"severity": level, "count": count}
            for level, count in severity_counts.items()
        ],
        "holdout_model_metrics": model_bundle.get("metrics", {}),
        "holdout_metrics_source": model_bundle.get("metrics", {}).get(
            "evaluation_dataset", "synthetic_demo"
        ),
        "production_labeled_metrics": production_metrics,
    }


@app.get("/api/transactions/recent")
def recent_transactions(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=200),
    suspicious_only: bool = False,
) -> dict[str, list[dict]]:
    with db:
        rows = db.scalars(
            select(Transaction).order_by(Transaction.timestamp.desc()).limit(limit)
        ).all()
        results = [_serialize(row) for row in rows]
    threshold = float(model_bundle["threshold"])
    suspicious = [row for row in results if row["fraud_score"] >= threshold]
    return {
        "transactions": suspicious if suspicious_only else results,
        "suspicious_transactions": suspicious,
    }


@app.get("/api/transactions/history")
def transaction_history(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    search: str | None = Query(default=None, max_length=128),
    transaction_status: str | None = Query(default=None, max_length=32),
    risk_level: Literal["Low", "Medium", "High", "Critical"] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    sort_by: Literal["timestamp", "amount", "fraud_score", "risk_level"] = "timestamp",
    sort_order: Literal["asc", "desc"] = "desc",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be before end_date")
    conditions = []
    if search:
        pattern = f"%{search.strip()}%"
        conditions.append(or_(
            Transaction.transaction_id.ilike(pattern),
            Transaction.user_id.ilike(pattern),
            Transaction.location.ilike(pattern),
            Transaction.merchant_category.ilike(pattern),
            Transaction.payment_method.ilike(pattern),
        ))
    if transaction_status:
        conditions.append(Transaction.status == transaction_status)
    if risk_level:
        conditions.append(Transaction.risk_level == risk_level)
    if start_date:
        conditions.append(Transaction.timestamp >= start_date)
    if end_date:
        conditions.append(Transaction.timestamp <= end_date)
    sort_column = {
        "timestamp": Transaction.timestamp,
        "amount": Transaction.amount,
        "fraud_score": Transaction.fraud_score,
        "risk_level": Transaction.risk_level,
    }[sort_by]
    ordering = sort_column.asc() if sort_order == "asc" else desc(sort_column)
    statement = select(Transaction)
    count_statement = select(func.count()).select_from(Transaction)
    if conditions:
        statement = statement.where(and_(*conditions))
        count_statement = count_statement.where(and_(*conditions))
    total = int(db.scalar(count_statement) or 0)
    rows = db.scalars(
        statement.order_by(ordering).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return {
        "items": [_serialize(row) for row in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": (total + page_size - 1) // page_size,
    }


@app.get("/api/alerts")
def list_alerts(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    search: str | None = Query(default=None, max_length=128),
    severity: Literal["Low", "Medium", "High", "Critical"] | None = None,
    alert_status: str | None = Query(default=None, max_length=32),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be before end_date")
    conditions = []
    if search:
        pattern = f"%{search.strip()}%"
        conditions.append(or_(
            Transaction.transaction_id.ilike(pattern),
            Transaction.user_id.ilike(pattern),
            Transaction.location.ilike(pattern),
            Alert.detection_reason.ilike(pattern),
            Alert.investigation_notes.ilike(pattern),
        ))
    if severity:
        conditions.append(Alert.severity == severity)
    if alert_status:
        conditions.append(Alert.status == alert_status)
    if start_date:
        conditions.append(Alert.created_at >= start_date)
    if end_date:
        conditions.append(Alert.created_at <= end_date)
    statement = select(Alert).join(Transaction, Alert.transaction_id == Transaction.transaction_id)
    count_statement = select(func.count()).select_from(Alert).join(
        Transaction, Alert.transaction_id == Transaction.transaction_id
    )
    if conditions:
        statement = statement.where(and_(*conditions))
        count_statement = count_statement.where(and_(*conditions))
    total = int(db.scalar(count_statement) or 0)
    alerts = db.scalars(
        statement.order_by(desc(Alert.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return {
        "items": [
            {
                **_serialize(alert.transaction),
                "alert_id": alert.id,
                "detection_time": alert.created_at.isoformat(),
                "severity": alert.severity,
                "investigation_status": alert.status,
                "detection_reason": json.loads(alert.detection_reason),
                "investigation_notes": alert.investigation_notes,
                "assigned_analyst": alert.assigned_analyst,
            }
            for alert in alerts
        ],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": (total + page_size - 1) // page_size,
    }


@app.post("/api/alerts/{transaction_id}/investigate")
def investigate_alert(
    transaction_id: str,
    payload: AlertInvestigationInput,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    alert = db.scalar(
        select(Alert).where(Alert.transaction_id == transaction_id)
    )
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    if payload.status == "Resolved" and user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required to resolve alerts")
    alert.status = payload.status
    alert.investigation_notes = payload.notes
    alert.assigned_analyst = alert.assigned_analyst or user.username
    alert.reviewed_at = datetime.now(timezone.utc)
    if payload.status == "Resolved":
        alert.resolved_at = datetime.now(timezone.utc)
    db.add(
        InvestigationRecord(
            transaction_id=transaction_id,
            alert_id=alert.id,
            analyst=user.username,
            action=payload.status,
            notes=payload.notes,
        )
    )
    _record_activity(
        db,
        user.username,
        "alert_investigation",
        f"Alert {transaction_id} moved to {payload.status}",
        details={"transaction_id": transaction_id},
    )
    db.commit()
    return {
        "transaction_id": transaction_id,
        "status": alert.status,
        "severity": alert.severity,
        "assigned_analyst": alert.assigned_analyst,
        "investigation_notes": alert.investigation_notes,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
    }


@app.post(
    "/api/alerts/{transaction_id}/action",
    response_model=TransactionOutput,
)
def take_alert_action(
    transaction_id: str,
    payload: ActionInput,
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> TransactionOutput:
    row = db.scalar(
        select(Transaction).where(Transaction.transaction_id == transaction_id)
    )
    if row is None or row.fraud_score < float(model_bundle["threshold"]):
        raise HTTPException(status_code=404, detail="Flagged transaction not found")
    row.status = payload.action
    if payload.action == "Reviewed/Genuine":
        row.actual_is_fraud = False
    if row.alert:
        row.alert.status = payload.action
        row.alert.reviewed_at = datetime.now(timezone.utc)
        row.alert.resolved_at = datetime.now(timezone.utc)
        row.alert.assigned_analyst = user.username
        db.add(
            InvestigationRecord(
                transaction_id=row.transaction_id,
                alert_id=row.alert.id,
                analyst=user.username,
                action=payload.action,
                notes="Action recorded from the alert review controls.",
            )
        )
    _record_activity(
        db,
        user.username,
        "alert_action",
        f"Alert {transaction_id} action: {payload.action}",
        details={"transaction_id": transaction_id, "action": payload.action},
    )
    db.commit()
    db.refresh(row)
    return TransactionOutput(**_serialize(row))


@app.post("/api/transactions/{transaction_id}/outcome")
def adjudicate_transaction(
    transaction_id: str,
    payload: TransactionOutcomeInput,
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    row = db.scalar(
        select(Transaction).where(Transaction.transaction_id == transaction_id)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    row.actual_is_fraud = payload.actual_is_fraud
    row.status = "Confirmed-Fraud" if payload.actual_is_fraud else "Reviewed/Genuine"
    if row.alert:
        row.alert.investigation_notes = payload.notes
        row.alert.assigned_analyst = user.username
        row.alert.reviewed_at = datetime.now(timezone.utc)
        if payload.actual_is_fraud:
            row.alert.status = "Confirmed-Fraud"
            row.alert.resolved_at = None
        else:
            row.alert.status = "Reviewed/Genuine"
            row.alert.resolved_at = datetime.now(timezone.utc)
    db.add(
        InvestigationRecord(
            transaction_id=row.transaction_id,
            alert_id=row.alert.id if row.alert else None,
            analyst=user.username,
            action="Confirmed-Fraud" if payload.actual_is_fraud else "Reviewed/Genuine",
            notes=payload.notes,
        )
    )
    _record_activity(
        db,
        user.username,
        "fraud_outcome_adjudication",
        f"Transaction {transaction_id} outcome recorded",
        severity="Warning" if payload.actual_is_fraud else "Info",
        details={"transaction_id": transaction_id, "actual_is_fraud": payload.actual_is_fraud},
    )
    db.commit()
    return {
        "transaction_id": transaction_id,
        "actual_is_fraud": row.actual_is_fraud,
        "status": row.status,
        "adjudicated_by": user.username,
    }


@app.get("/api/transactions/{transaction_id}/history")
def transaction_investigation_history(
    transaction_id: str,
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    if db.scalar(
        select(Transaction.id).where(Transaction.transaction_id == transaction_id)
    ) is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    records = db.scalars(
        select(InvestigationRecord)
        .where(InvestigationRecord.transaction_id == transaction_id)
        .order_by(InvestigationRecord.created_at)
    ).all()
    return [
        {
            "analyst": record.analyst,
            "action": record.action,
            "notes": record.notes,
            "created_at": record.created_at.isoformat(),
        }
        for record in records
    ]


def _serialize_incident(incident: Incident) -> dict:
    return {
        "incident_id": incident.incident_id,
        "description": incident.description,
        "severity": incident.severity,
        "assigned_analyst": incident.assigned_analyst,
        "investigation_notes": incident.investigation_notes,
        "status": incident.status,
        "resolution_details": incident.resolution_details,
        "created_at": incident.created_at.isoformat(),
        "updated_at": incident.updated_at.isoformat(),
        "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
    }


@app.get("/api/incidents")
def list_incidents(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    search: str | None = Query(default=None, max_length=128),
    severity: Literal["Low", "Medium", "High", "Critical"] | None = None,
    incident_status: Literal[
        "Triage", "Contained", "Evidence Preserved", "Recovery", "Resolved"
    ] | None = None,
) -> list[dict]:
    statement = select(Incident)
    conditions = []
    if search:
        pattern = f"%{search.strip()}%"
        conditions.append(or_(
            Incident.incident_id.ilike(pattern),
            Incident.description.ilike(pattern),
            Incident.assigned_analyst.ilike(pattern),
        ))
    if severity:
        conditions.append(Incident.severity == severity)
    if incident_status:
        conditions.append(Incident.status == incident_status)
    if conditions:
        statement = statement.where(and_(*conditions))
    incidents = db.scalars(statement.order_by(desc(Incident.created_at))).all()
    return [_serialize_incident(incident) for incident in incidents]


@app.post("/api/incidents", status_code=status.HTTP_201_CREATED)
def create_incident(
    payload: IncidentInput,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    incident = Incident(
        incident_id=f"INC-{uuid4().hex[:12].upper()}",
        description=payload.description,
        severity=payload.severity,
        assigned_analyst=payload.assigned_analyst or user.username,
    )
    db.add(incident)
    _record_activity(
        db,
        user.username,
        "incident_created",
        f"Incident {incident.incident_id} created",
        severity=payload.severity,
        details={"incident_id": incident.incident_id, "severity": payload.severity},
    )
    db.flush()
    db.add(
        InvestigationRecord(
            incident_id=incident.incident_id,
            analyst=user.username,
            action="Triage",
            notes="Incident created and triaged.",
        )
    )
    db.commit()
    db.refresh(incident)
    return _serialize_incident(incident)


@app.get("/api/incidents/{incident_id}/history")
def incident_history(
    incident_id: str,
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    if db.scalar(select(Incident.id).where(Incident.incident_id == incident_id)) is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    records = db.scalars(
        select(InvestigationRecord)
        .where(InvestigationRecord.incident_id == incident_id)
        .order_by(InvestigationRecord.created_at)
    ).all()
    return [
        {
            "analyst": record.analyst,
            "action": record.action,
            "notes": record.notes,
            "created_at": record.created_at.isoformat(),
        }
        for record in records
    ]


@app.patch("/api/incidents/{incident_id}")
def update_incident(
    incident_id: str,
    payload: IncidentUpdateInput,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    incident = db.scalar(
        select(Incident).where(Incident.incident_id == incident_id)
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="At least one incident field is required")
    if changes.get("status") == "Resolved" and user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required to resolve incidents")
    for field_name, value in changes.items():
        setattr(incident, field_name, value)
    if incident.status == "Resolved":
        incident.resolved_at = datetime.now(timezone.utc)
    elif "status" in changes:
        incident.resolved_at = None
    note = changes.get("investigation_notes") or changes.get("resolution_details") or ""
    db.add(
        InvestigationRecord(
            incident_id=incident.incident_id,
            analyst=user.username,
            action=changes.get("status", "Updated"),
            notes=note,
        )
    )
    _record_activity(
        db,
        user.username,
        "incident_updated",
        f"Incident {incident.incident_id} updated",
        severity=incident.severity,
        details={"incident_id": incident.incident_id, "status": incident.status},
    )
    db.commit()
    db.refresh(incident)
    return _serialize_incident(incident)


@app.get("/api/alerts/{transaction_id}/history")
def alert_history(
    transaction_id: str,
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    alert = db.scalar(
        select(Alert).where(Alert.transaction_id == transaction_id)
    )
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    records = db.scalars(
        select(InvestigationRecord)
        .where(InvestigationRecord.alert_id == alert.id)
        .order_by(InvestigationRecord.created_at)
    ).all()
    return [
        {
            "analyst": record.analyst,
            "action": record.action,
            "notes": record.notes,
            "created_at": record.created_at.isoformat(),
        }
        for record in records
    ]


@app.get("/api/activity")
def list_user_activity(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    statement = select(UserActivity)
    if user.role != "admin":
        statement = statement.where(UserActivity.username == user.username)
    activities = db.scalars(
        statement.order_by(desc(UserActivity.created_at)).limit(limit)
    ).all()
    return [
        {
            "id": activity.id,
            "username": activity.username,
            "event_type": activity.event_type,
            "description": activity.description,
            "details": json.loads(activity.details),
            "created_at": activity.created_at.isoformat(),
        }
        for activity in activities
    ]


@app.get("/api/security/events")
def list_security_events(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    severity: str | None = Query(default=None, max_length=16),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    statement = select(SecurityLog)
    if severity:
        statement = statement.where(SecurityLog.severity == severity)
    events = db.scalars(
        statement.order_by(desc(SecurityLog.created_at)).limit(limit)
    ).all()
    return [
        {
            "id": event.id,
            "username": event.username,
            "event_type": event.event_type,
            "severity": event.severity,
            "summary": event.summary,
            "details": json.loads(event.details),
            "created_at": event.created_at.isoformat(),
        }
        for event in events
    ]


@app.get("/api/users")
def list_users(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    users = db.scalars(select(User).order_by(User.username)).all()
    return [
        {"username": user.username, "role": user.role, "is_active": user.is_active}
        for user in users
    ]


@app.patch("/api/users/{username}")
def update_user(
    username: str,
    payload: UserAdminUpdateInput,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    changes = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if value is not None
    }
    if not changes:
        raise HTTPException(status_code=422, detail="At least one user field is required")
    if user.id == current_user.id and changes.get("is_active") is False:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
    if user.role == "admin" and user.is_active and (
        changes.get("role") == "analyst" or changes.get("is_active") is False
    ):
        active_admins = db.scalar(
            select(func.count()).select_from(User).where(
                User.role == "admin",
                User.is_active.is_(True),
                User.id != user.id,
            )
        )
        if not active_admins:
            raise HTTPException(status_code=400, detail="At least one active administrator is required")
    for field_name, value in changes.items():
        setattr(user, field_name, value)
    if changes.get("is_active") is False or changes.get("role") is not None:
        user.token_version += 1
    _record_activity(
        db,
        current_user.username,
        "user_admin_update",
        f"Account {username} updated",
        severity="Warning",
        details={"username": username, **changes},
    )
    db.commit()
    return {"username": user.username, "role": user.role, "is_active": user.is_active}


@app.delete("/api/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    username: str,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    if user.role == "admin" and user.is_active:
        active_admins = db.scalar(
            select(func.count()).select_from(User).where(
                User.role == "admin",
                User.is_active.is_(True),
                User.id != user.id,
            )
        )
        if not active_admins:
            raise HTTPException(
                status_code=400,
                detail="At least one active administrator is required",
            )
    _record_activity(
        db,
        current_user.username,
        "user_admin_delete",
        f"Account {username} deleted",
        severity="Warning",
        details={"username": username, "role": user.role, "was_active": user.is_active},
    )
    db.delete(user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/testing/run")
def run_security_checks(
    user: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    checks: list[dict[str, str]] = []

    def add_check(
        case_id: str,
        description: str,
        expected: str,
        actual: str,
        passed: bool,
        input_data: dict | None = None,
    ) -> None:
        result = "PASS" if passed else "FAIL"
        checks.append(
            {
                "case_id": case_id,
                "description": description,
                "expected_result": expected,
                "actual_result": actual,
                "status": result,
            }
        )
        db.add(
            TestResult(
                case_id=case_id,
                description=description,
                input_data=json.dumps(input_data or {}),
                expected_result=expected,
                actual_result=actual,
                status=result,
            )
        )

    database_value = db.execute(text("SELECT 1")).scalar_one()
    add_check(
        "APP-DB-001",
        "Database connectivity",
        "SELECT 1 returns 1",
        f"SELECT 1 returned {database_value}",
        database_value == 1,
    )
    threshold = float(model_bundle["threshold"])
    add_check(
        "ML-ART-001",
        "Model and decision threshold loaded",
        "Model artifact present and threshold within [0, 1]",
        f"Loaded threshold {threshold:.4f}",
        0 <= threshold <= 1 and "model" in model_bundle,
    )
    sample = TransactionInput(
        user_id="TEST-USER",
        amount=125.0,
        location="New York",
        merchant_category="Retail",
    )
    sample_features = prepare_features(pd.DataFrame([sample.model_dump()]))
    probability = float(model_bundle["model"].predict_proba(sample_features)[0, 1])
    anomaly_features = model_bundle["anomaly_preprocessor"].transform(sample_features)
    anomaly_detector_output = int(
        model_bundle["anomaly_detector"].predict(anomaly_features)[0]
    )
    add_check(
        "ML-INF-001",
        "Model inference shape and probability",
        "One finite fraud probability between 0 and 1",
        f"Probability {probability:.6f}",
        0 <= probability <= 1,
        {"amount": 125.0, "location": "New York", "merchant_category": "Retail"},
    )
    add_check(
        "ML-ANOM-001",
        "Anomaly detector inference",
        "Isolation Forest returns -1 or 1",
        f"Detector returned {anomaly_detector_output}",
        anomaly_detector_output in (-1, 1),
    )
    validation_passed = False
    validation_actual = "Invalid amount was accepted"
    try:
        TransactionInput(
            user_id="TEST-USER",
            amount=0,
            location="New York",
            merchant_category="Retail",
        )
    except ValidationError:
        validation_passed = True
        validation_actual = "Zero amount was rejected by request validation"
    add_check(
        "API-VAL-001",
        "Transaction input validation",
        "Zero amount is rejected",
        validation_actual,
        validation_passed,
        {"amount": 0},
    )
    add_check(
        "AUTH-RBAC-001",
        "Administrative security-test authorization",
        "Only an active admin may run checks",
        f"Authorized caller role is {user.role}",
        user.role == "admin" and user.is_active,
    )
    _record_activity(
        db,
        user.username,
        "security_tests_run",
        "Application security and ML checks executed",
        severity="Warning",
        details={"case_count": len(checks)},
    )
    db.commit()
    return {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "passed": sum(case["status"] == "PASS" for case in checks),
        "failed": sum(case["status"] == "FAIL" for case in checks),
        "results": checks,
    }


@app.get("/api/testing/results")
def list_test_results(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    results = db.scalars(
        select(TestResult).order_by(desc(TestResult.executed_at)).limit(limit)
    ).all()
    return [
        {
            "id": result.id,
            "case_id": result.case_id,
            "description": result.description,
            "input": json.loads(result.input_data),
            "expected_result": result.expected_result,
            "actual_result": result.actual_result,
            "status": result.status,
            "executed_at": result.executed_at.isoformat(),
        }
        for result in results
    ]


@app.get("/api/reports/preview")
def report_preview(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    report: Literal[
        "transaction_summary",
        "fraud_detection",
        "security_monitoring",
        "incident_response",
        "ml_model_performance",
        "security_testing",
    ] = "transaction_summary",
) -> dict:
    titles = {
        "transaction_summary": "Transaction Summary Report",
        "fraud_detection": "Fraud Detection Report",
        "security_monitoring": "Security Monitoring Report",
        "incident_response": "Incident Response Report",
        "ml_model_performance": "ML Model Performance Report",
        "security_testing": "Security Testing Report",
    }
    summary: dict = {}
    if report == "transaction_summary":
        rows = db.scalars(
            select(Transaction).order_by(desc(Transaction.timestamp)).limit(10)
        ).all()
        total = int(db.scalar(select(func.count()).select_from(Transaction)) or 0)
        summary = {
            "total_transactions": total,
            "total_value": round(float(db.scalar(select(func.sum(Transaction.amount))) or 0), 2),
            "flagged_transactions": int(db.scalar(
                select(func.count()).select_from(Transaction).where(
                    Transaction.fraud_score >= float(model_bundle["threshold"])
                )
            ) or 0),
        }
        records = [_serialize(row) for row in rows]
    elif report == "fraud_detection":
        statement = select(Transaction).where(
            Transaction.fraud_score >= float(model_bundle["threshold"])
        )
        total = int(db.scalar(
            select(func.count()).select_from(Transaction).where(
                Transaction.fraud_score >= float(model_bundle["threshold"])
            )
        ) or 0)
        rows = db.scalars(statement.order_by(desc(Transaction.timestamp)).limit(10)).all()
        summary = {
            "flagged_transactions": total,
            "threshold": float(model_bundle["threshold"]),
            "flagged_value": round(float(db.scalar(
                select(func.sum(Transaction.amount)).where(
                    Transaction.fraud_score >= float(model_bundle["threshold"])
                )
            ) or 0), 2),
        }
        records = [_serialize(row) for row in rows]
    elif report == "security_monitoring":
        total = int(db.scalar(select(func.count()).select_from(SecurityLog)) or 0)
        rows = db.scalars(
            select(SecurityLog).order_by(desc(SecurityLog.created_at)).limit(10)
        ).all()
        summary = {"security_events": total}
        records = [
            {
                "event_type": row.event_type,
                "severity": row.severity,
                "summary": row.summary,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    elif report == "incident_response":
        incidents = db.scalars(
            select(Incident).order_by(desc(Incident.created_at)).limit(10)
        ).all()
        total = int(db.scalar(select(func.count()).select_from(Incident)) or 0)
        summary = {"incidents": total}
        records = [_serialize_incident(row) for row in incidents]
    elif report == "ml_model_performance":
        summary = {
            **model_bundle.get("metrics", {}),
            "evaluation_dataset": model_bundle.get("metrics", {}).get(
                "evaluation_dataset", "synthetic_demo"
            ),
        }
        records = []
    else:
        results = db.scalars(
            select(TestResult).order_by(desc(TestResult.executed_at)).limit(10)
        ).all()
        total = int(db.scalar(select(func.count()).select_from(TestResult)) or 0)
        passed = int(db.scalar(
            select(func.count()).select_from(TestResult).where(TestResult.status == "PASS")
        ) or 0)
        summary = {"executed_test_results": total, "passed_test_results": passed}
        records = [
            {
                "case_id": row.case_id,
                "description": row.description,
                "expected_result": row.expected_result,
                "actual_result": row.actual_result,
                "status": row.status,
                "executed_at": row.executed_at.isoformat(),
            }
            for row in results
        ]
    return {
        "report": report,
        "title": titles[report],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "records": records,
        "preview_limit": 10,
    }


@app.get("/api/reports/export")
def export_report(
    _: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    report: Literal[
        "transaction_summary",
        "fraud_detection",
        "security_monitoring",
        "incident_response",
        "ml_model_performance",
        "security_testing",
    ] = "transaction_summary",
) -> StreamingResponse:
    if report in {"transaction_summary", "fraud_detection"}:
        statement = select(Transaction).order_by(desc(Transaction.timestamp)).limit(5000)
        if report == "fraud_detection":
            statement = statement.where(
                Transaction.fraud_score >= float(model_bundle["threshold"])
            )
        rows = [_serialize(row) for row in db.scalars(statement).all()]
    elif report == "security_monitoring":
        rows = [
            {
                "username": row.username,
                "event_type": row.event_type,
                "severity": row.severity,
                "summary": row.summary,
                "created_at": row.created_at.isoformat(),
            }
            for row in db.scalars(
                select(SecurityLog).order_by(desc(SecurityLog.created_at)).limit(5000)
            ).all()
        ]
    elif report == "incident_response":
        rows = [
            _serialize_incident(row)
            for row in db.scalars(
                select(Incident).order_by(desc(Incident.created_at)).limit(5000)
            ).all()
        ]
    elif report == "ml_model_performance":
        rows = [model_bundle.get("metrics", {})]
    else:
        rows = [
            {
                "case_id": row.case_id,
                "description": row.description,
                "expected_result": row.expected_result,
                "actual_result": row.actual_result,
                "status": row.status,
                "executed_at": row.executed_at.isoformat(),
            }
            for row in db.scalars(
                select(TestResult).order_by(desc(TestResult.executed_at)).limit(5000)
            ).all()
        ]
    output = io.StringIO()
    if rows:
        fieldnames = list(dict.fromkeys(key for item in rows for key in item))
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, default=str)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    else:
        output.write("No records found\n")
    filename = f"frauddetect-{report}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws/transactions")
async def transaction_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        credentials = await websocket.receive_json()
        if not isinstance(credentials, dict):
            await websocket.close(code=4401, reason="Invalid authentication")
            return
        payload = _decode_token(str(credentials.get("token", "")))
        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.username == payload["sub"]))
            if (
                user is None
                or not user.is_active
                or payload.get("ver") != user.token_version
            ):
                await websocket.close(code=4401, reason="Invalid authentication")
                return
        manager.connections.add(websocket)
        await websocket.send_json({"type": "authenticated"})
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, ValueError, TypeError):
        manager.disconnect(websocket)
    except HTTPException:
        manager.disconnect(websocket)
        await websocket.close(code=4401, reason="Invalid or expired authentication")
