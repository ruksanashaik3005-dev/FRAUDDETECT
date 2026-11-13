# FRAUDDETECT Threat Model

## Scope and trust boundaries

This model covers the local/Compose demo: React browser client, FastAPI API,
SQLite or PostgreSQL persistence, Random Forest fraud scoring, Isolation Forest
anomaly signals, WebSocket streaming, CSV report exports, and the admin/analyst
workflow. The system uses synthetic training data and is not a payment
authorization service.

Trust boundaries:

1. Browser to API: bearer-token authentication; transaction analysis also
   requires a valid server-side session.
2. Service-to-service ingestion to API: `X-API-Key` on the ingestion route.
3. API to database: SQLAlchemy parameterized statements and deployment
   connection settings.
4. API to model artifact: local trusted artifact loaded by the backend process.
5. Compose host to services: only the frontend port is host-published by
   default; the database and backend use the internal bridge network.

## Assets

- Usernames, password hashes, role assignments, and active-session versions.
- Transaction amounts, account-balance/device fields, model scores, reasons,
  alerts, investigation notes, and adjudicated outcomes.
- JWT signing secret, ingestion API key, database credentials, and model files.
- Security logs, user activity, incident history, and test results.

## Threats and controls

| Threat | Control in the demo | Residual risk |
|---|---|---|
| Password disclosure | Bcrypt password hashes; password values are not logged or returned. | No MFA, password reset, or credential-stuffing rate limiter. |
| Unauthorized admin actions | Server-side role checks; analyst accounts cannot self-select admin; admin role changes revoke prior tokens. | Initial admin setup and secrets remain operator responsibilities. |
| Replayed/stale JWT | Expiration plus persisted per-user token version; logout increments the version. | Token storage in browser `sessionStorage` remains exposed to same-origin script compromise. |
| Forged ingestion | Constant-time `X-API-Key` comparison on server-to-server evaluation. | Shared API-key rotation and ingestion rate limiting are not implemented. |
| Invalid/malicious input | Pydantic bounds and SQLAlchemy query construction. | Business rules and production-scale abuse limits need further validation. |
| Data tampering or repudiation | User activity and security logs, fraud prediction rows, incident and investigation history. | Logs are in the application database, not cryptographically immutable or externally archived. |
| Sensitive-data exposure | Authenticated API routes; no passwords, bearer tokens, or API keys in event details or frontend bundles. | Analyst access is broad in this demo; production requires tenant scoping, masking, retention controls, and audited export permissions. |
| Data drift/model degradation | `monitor_drift.py` compares distributions; production performance is calculated only with adjudicated labels. | Synthetic holdout performance is not representative; no automatic retraining or model promotion occurs. |
| Network exposure | Compose does not publish PostgreSQL or backend ports by default; local port binds to loopback. | TLS must be terminated by a trusted ingress before external exposure. |
| Unsafe lab activity | The application does not scan networks or capture traffic. | Operators must independently enforce lab authorization and privacy. |

## Data handling

Application logs deliberately exclude passwords, bearer tokens, and API keys.
Transaction details and investigation notes are persisted in the configured
database. Configure access control, backups, retention, encryption at rest, and
deletion policies for any non-demo deployment. Synthetic training and sample
lab content must not be presented as evidence of real-world detections.

## Out of scope

Payment settlement, chargeback integrations, customer identity verification,
MFA, external threat-intelligence feeds, tamper-proof audit storage, and
automated containment of real payment instruments are not implemented.
