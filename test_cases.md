# FRAUDDETECT Test Cases

## Automated integration suite

Run with:

```powershell
python -m unittest discover -s tests -v
```

`tests/test_api_workflows.py` creates an isolated in-memory SQLite database and
executes the FastAPI lifespan and endpoints. The suite contains an end-to-end
API workflow test and a legacy-schema migration test. It does not modify the
local demo database.

| Case ID | Test | Input | Expected result |
|---|---|---|---|
| AUTH-001 | Protected dashboard requires login | Request stats without a bearer token | HTTP 401 |
| AUTH-002 | Analyst registration cannot set role | Register with a requested `admin` role | HTTP 201 and role `analyst` |
| AUTH-003 | Password length validation | Register with a short password | HTTP 422 |
| ML-001 | Transaction scoring and persistence | High-frequency, VPN, high-amount synthetic demo transaction | Prediction, score, risk level, signals, and saved records |
| API-KEY-001 | Ingestion key required | Call ingestion without and with `X-API-Key` | HTTP 401 without; successful evaluation with valid test key |
| RBAC-001 | Admin-only alert action | Analyst and admin submit Block action | Analyst HTTP 403; admin action succeeds |
| ALERT-001 | Alert investigation history | Start investigation and request history | Notes, analyst, action, and timestamp are returned |
| OUTCOME-001 | Outcome adjudication | Admin records a confirmed fraud outcome | Outcome is persisted and alert becomes Confirmed-Fraud |
| USER-001 | User administration | Admin lists users and attempts to demote the last admin | User list succeeds; last-admin demotion returns HTTP 400 |
| HISTORY-001 | Transaction search and pagination | Search by transaction ID | Matching record and pagination metadata |
| ANALYTICS-001 | Analytics use database rows | Query current 30-day window | Genuine/fraud counts match test transactions |
| ANALYTICS-002 | Labeled production metrics | Adjudicate a test transaction and query analytics | Labeled sample count reflects adjudicated outcomes |
| INCIDENT-001 | Incident lifecycle and audit history | Create, update, and inspect incident history | Persisted status and investigation event |
| REPORT-001 | Incident report export | Export incident response report | CSV response |
| REPORT-002 | Report preview | Preview transaction summary | Summary reflects persisted transactions |
| TEST-001 | Safe security/model checks | Admin executes `/api/testing/run` | Six checks pass and are persisted |
| LOGOUT-001 | Server-side session revocation | Logout then reuse token | Logout succeeds; subsequent protected request is HTTP 401 |
| MIGRATION-001 | Legacy SQLite migration | Upgrade a database with pre-existing records | New columns and derived fields are added; existing user, transaction, and alert data remain |

## Run-time checks

The admin-only **Security Testing** page executes database connectivity,
artifact/threshold integrity, model inference, anomaly-detector inference,
request validation, and administrative role-gate checks. These are
non-destructive checks; model inference uses an in-memory sample and does not
create a transaction record.

## Manual checks still required for deployment

- Verify PostgreSQL migration with a backup of the target database.
- Validate TLS termination, production CORS origins, secret rotation, backups,
  database retention, and audit-log archival.
- Verify responsive dashboard behavior in the supported target browsers.
- Validate thresholds and model behavior using consented, adjudicated data.
- Verify Docker Compose deployment on a host with Docker available.
