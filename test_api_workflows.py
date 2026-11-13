import os
import json
import unittest

import numpy as np
import pandas as pd

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["FRAUDDETECT_JWT_SECRET"] = "unit-test-jwt-secret-at-least-32-characters"
os.environ["FRAUDDETECT_API_KEY"] = "unit-test-api-key-long-enough-for-tests"
os.environ["FRAUDDETECT_ADMIN_USERNAME"] = "admin"
os.environ["FRAUDDETECT_ADMIN_PASSWORD"] = "LocalTestAdminPassword!2026"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.pool import StaticPool

import main
from ml_pipeline import prepare_features
from models import FraudPrediction, Transaction


class APIWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.engine.dispose()
        main.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        main.SessionLocal.configure(bind=main.engine)
        cls.client_context = TestClient(main.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        main.engine.dispose()

    def test_cors_allows_admin_user_updates(self):
        for method in ("PATCH", "DELETE"):
            with self.subTest(method=method):
                response = self.client.options(
                    "/api/users/analyst_one",
                    headers={
                        "Origin": "http://127.0.0.1:5173",
                        "Access-Control-Request-Method": method,
                        "Access-Control-Request-Headers": "authorization,content-type",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn(
                    method,
                    response.headers["access-control-allow-methods"],
                )

    def test_admin_can_disable_activate_and_delete_user(self):
        client = self.client
        admin_login = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": os.environ["FRAUDDETECT_ADMIN_PASSWORD"]},
        )
        admin_headers = {
            "Authorization": f"Bearer {admin_login.json()['access_token']}"
        }
        username = "account_lifecycle_test"
        registration = client.post(
            "/api/auth/register",
            json={"username": username, "password": "LifecyclePassword!2026"},
        )
        self.assertEqual(registration.status_code, 201, registration.text)

        disabled = client.patch(
            f"/api/users/{username}",
            headers=admin_headers,
            json={"is_active": False},
        )
        self.assertEqual(disabled.status_code, 200, disabled.text)
        self.assertFalse(disabled.json()["is_active"])
        self.assertEqual(
            client.post(
                "/api/auth/login",
                data={"username": username, "password": "LifecyclePassword!2026"},
            ).status_code,
            401,
        )

        activated = client.patch(
            f"/api/users/{username}",
            headers=admin_headers,
            json={"is_active": True},
        )
        self.assertEqual(activated.status_code, 200, activated.text)
        self.assertTrue(activated.json()["is_active"])
        self.assertEqual(
            client.post(
                "/api/auth/login",
                data={"username": username, "password": "LifecyclePassword!2026"},
            ).status_code,
            200,
        )

        deleted = client.delete(f"/api/users/{username}", headers=admin_headers)
        self.assertEqual(deleted.status_code, 204, deleted.text)
        self.assertNotIn(
            username,
            {user["username"] for user in client.get("/api/users", headers=admin_headers).json()},
        )
        activity = client.get("/api/activity", headers=admin_headers).json()
        self.assertTrue(any(
            item["description"] == f"Account {username} deleted"
            for item in activity
        ))
        self.assertEqual(
            client.delete("/api/users/admin", headers=admin_headers).status_code,
            400,
        )

    def test_auth_transaction_alert_incident_and_reports(self):
        client = self.client
        holdout = main.model_bundle["metrics"]
        tn, fp, fn, tp = np.asarray(holdout["confusion_matrix"]).ravel()
        self.assertAlmostEqual(
            holdout["false_positive_rate"], fp / (fp + tn)
        )
        self.assertAlmostEqual(
            holdout["false_negative_rate"], fn / (fn + tp)
        )
        missing_features = prepare_features(pd.DataFrame([{
            "amount": np.nan,
            "timestamp": "2026-11-12T08:00:00Z",
            "location": "New York",
            "merchant_category": "Retail",
            "is_vpn": 0,
            "transaction_frequency_1h": 0,
        }]))
        self.assertTrue(
            np.isfinite(main.model_bundle["model"].predict_proba(missing_features)).all()
        )
        self.assertEqual(client.get("/api/dashboard/stats").status_code, 401)

        admin_login = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": os.environ["FRAUDDETECT_ADMIN_PASSWORD"]},
        )
        self.assertEqual(admin_login.status_code, 200, admin_login.text)
        admin_token = admin_login.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        registration = client.post(
            "/api/auth/register",
            json={
                "username": "analyst_one",
                "password": "AnalystPassword!2026",
                "role": "admin",
            },
        )
        self.assertEqual(registration.status_code, 201, registration.text)
        self.assertEqual(registration.json()["role"], "analyst")
        self.assertEqual(
            client.post(
                "/api/auth/register",
                json={"username": "short", "password": "short"},
            ).status_code,
            422,
        )
        analyst_login = client.post(
            "/api/auth/login",
            data={"username": "analyst_one", "password": "AnalystPassword!2026"},
        )
        analyst_headers = {
            "Authorization": f"Bearer {analyst_login.json()['access_token']}"
        }

        payload = {
            "transaction_id": "TX-TEST-001",
            "user_id": "USR-TEST",
            "amount": 4200,
            "payment_method": "Card",
            "timestamp": "2026-11-12T08:00:00Z",
            "account_balance": 1000,
            "location": "Lagos",
            "device_information": "test browser",
            "merchant_category": "Electronics",
            "is_vpn": True,
            "transaction_frequency_1h": 8,
        }
        analyzed = client.post(
            "/api/transactions/analyze", headers=admin_headers, json=payload
        )
        self.assertEqual(analyzed.status_code, 200, analyzed.text)
        transaction = analyzed.json()
        self.assertEqual(transaction["payment_method"], "Card")
        self.assertEqual(transaction["device_information"], "test browser")
        self.assertEqual(transaction["prediction"], "Fraudulent")
        self.assertGreaterEqual(transaction["fraud_score"], main.model_bundle["threshold"])
        self.assertTrue(transaction["suspicious_features"])

        self.assertEqual(
            client.post("/api/transactions/evaluate", json=payload).status_code, 401
        )
        ingest_payload = {**payload, "transaction_id": "TX-TEST-APIKEY-01"}
        ingest = client.post(
            "/api/transactions/evaluate",
            headers={"X-API-Key": os.environ["FRAUDDETECT_API_KEY"]},
            json=ingest_payload,
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)

        alerts = client.get("/api/alerts?severity=Critical", headers=admin_headers)
        self.assertEqual(alerts.status_code, 200, alerts.text)
        self.assertGreaterEqual(alerts.json()["total"], 1)
        self.assertEqual(
            client.post(
                "/api/alerts/TX-TEST-001/action",
                headers=analyst_headers,
                json={"action": "Blocked"},
            ).status_code,
            403,
        )
        investigation = client.post(
            "/api/alerts/TX-TEST-001/investigate",
            headers=analyst_headers,
            json={"status": "Investigating", "notes": "Reviewing velocity and location."},
        )
        self.assertEqual(investigation.status_code, 200, investigation.text)
        self.assertTrue(
            client.get(
                "/api/alerts/TX-TEST-001/history", headers=admin_headers
            ).json()
        )
        self.assertEqual(
            client.post(
                "/api/alerts/TX-TEST-001/action",
                headers=admin_headers,
                json={"action": "Blocked"},
            ).status_code,
            200,
        )
        outcome = client.post(
            "/api/transactions/TX-TEST-001/outcome",
            headers=admin_headers,
            json={"actual_is_fraud": True, "notes": "Confirmed in test adjudication."},
        )
        self.assertEqual(outcome.status_code, 200, outcome.text)
        self.assertTrue(outcome.json()["actual_is_fraud"])
        self.assertEqual(
            client.get(
                "/api/alerts?alert_status=Confirmed-Fraud",
                headers=admin_headers,
            ).json()["total"],
            1,
        )
        self.assertGreaterEqual(
            len(client.get("/api/users", headers=admin_headers).json()), 2
        )
        self.assertEqual(
            client.patch(
                "/api/users/admin",
                headers=admin_headers,
                json={"role": "analyst"},
            ).status_code,
            400,
        )

        history = client.get(
            "/api/transactions/history?search=TX-TEST-001&page_size=5"
            "&start_date=2026-11-11T00:00:00Z",
            headers=admin_headers,
        )
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json()["total"], 1)
        analytics = client.get(
            "/api/dashboard/analytics?days=30", headers=admin_headers
        )
        self.assertEqual(analytics.status_code, 200, analytics.text)
        self.assertEqual(
            sum(row["value"] for row in analytics.json()["genuine_vs_fraudulent"]),
            2,
        )
        self.assertEqual(
            analytics.json()["production_labeled_metrics"]["labeled_samples"], 1
        )

        incident = client.post(
            "/api/incidents",
            headers=admin_headers,
            json={"description": "Suspicious payment activity in test", "severity": "High"},
        )
        self.assertEqual(incident.status_code, 201, incident.text)
        incident_id = incident.json()["incident_id"]
        self.assertEqual(
            client.patch(
                f"/api/incidents/{incident_id}",
                headers=admin_headers,
                json={
                    "status": "Evidence Preserved",
                    "investigation_notes": "Transaction and alert records retained.",
                },
            ).status_code,
            200,
        )
        self.assertTrue(
            client.get(
                f"/api/incidents/{incident_id}/history", headers=admin_headers
            ).json()
        )
        self.assertIn(
            "text/csv",
            client.get(
                "/api/reports/export?report=incident_response",
                headers=admin_headers,
            ).headers["content-type"],
        )
        preview = client.get(
            "/api/reports/preview?report=transaction_summary",
            headers=admin_headers,
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["summary"]["total_transactions"], 2)

        test_run = client.post("/api/testing/run", headers=admin_headers)
        self.assertEqual(test_run.status_code, 200, test_run.text)
        self.assertEqual(test_run.json()["failed"], 0)
        with main.SessionLocal() as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(Transaction)), 2)
            self.assertEqual(
                db.scalar(select(func.count()).select_from(FraudPrediction)), 2
            )

        self.assertEqual(
            client.post("/api/auth/logout", headers=admin_headers).status_code, 200
        )
        self.assertEqual(
            client.get("/api/dashboard/stats", headers=admin_headers).status_code, 401
        )

    def test_legacy_sqlite_schema_migration_preserves_records(self):
        previous_engine = main.engine
        legacy_engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        try:
            with legacy_engine.begin() as connection:
                connection.execute(text(
                    "CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(64) "
                    "UNIQUE NOT NULL, hashed_password VARCHAR(255) NOT NULL, role VARCHAR(16) NOT NULL)"
                ))
                connection.execute(text(
                    "CREATE TABLE transactions (id INTEGER PRIMARY KEY, transaction_id VARCHAR(64) "
                    "UNIQUE NOT NULL, user_id VARCHAR(64) NOT NULL, amount FLOAT NOT NULL, "
                    "timestamp DATETIME NOT NULL, location VARCHAR(100) NOT NULL, "
                    "merchant_category VARCHAR(80) NOT NULL, is_vpn BOOLEAN NOT NULL, "
                    "transaction_frequency_1h INTEGER NOT NULL, fraud_score FLOAT NOT NULL, "
                    "prediction VARCHAR(24) NOT NULL, status VARCHAR(32) NOT NULL, "
                    "reason_codes TEXT NOT NULL)"
                ))
                connection.execute(text(
                    "CREATE TABLE alerts (id INTEGER PRIMARY KEY, transaction_id VARCHAR(64) "
                    "NOT NULL UNIQUE, priority VARCHAR(16) NOT NULL, status VARCHAR(24) NOT NULL, "
                    "created_at DATETIME NOT NULL, reviewed_at DATETIME)"
                ))
                connection.execute(text(
                    "INSERT INTO users VALUES (1, 'existing_admin', 'hash', 'admin')"
                ))
                connection.execute(text(
                    "INSERT INTO transactions VALUES "
                    "(1, 'TX-LEGACY-001', 'USR-OLD', 500, '2026-11-09 10:00:00', "
                    "'Lagos', 'Retail', 1, 4, 0.96, 'Fraudulent', 'Flagged-Fraudulent', "
                    "'[\"VPN Detected\"]')"
                ))
                connection.execute(text(
                    "INSERT INTO alerts VALUES "
                    "(1, 'TX-LEGACY-001', 'High', 'Open', '2026-11-09 10:01:00', NULL)"
                ))
            main.engine = legacy_engine
            main.Base.metadata.create_all(bind=legacy_engine)
            main._migrate_inference_columns()
            columns = {
                column["name"] for column in inspect(legacy_engine).get_columns("transactions")
            }
            self.assertIn("device_information", columns)
            self.assertIn("risk_level", columns)
            with legacy_engine.connect() as connection:
                row = connection.execute(text(
                    "SELECT amount_to_user_avg_ratio, is_high_risk_country_and_vpn, "
                    "velocity_amount_product, hour, day_of_week, risk_level "
                    "FROM transactions WHERE transaction_id = 'TX-LEGACY-001'"
                )).one()
                self.assertEqual(tuple(row), (1.0, 1, 2000.0, 10, 0, "Critical"))
                alert = connection.execute(text(
                    "SELECT severity, detection_reason FROM alerts WHERE id = 1"
                )).one()
                self.assertEqual(alert.severity, "Critical")
                self.assertEqual(json.loads(alert.detection_reason), ["VPN Detected"])
                self.assertEqual(
                    connection.execute(text("SELECT COUNT(*) FROM users")).scalar_one(),
                    1,
                )
        finally:
            main.engine = previous_engine
            main.SessionLocal.configure(bind=previous_engine)
            legacy_engine.dispose()


if __name__ == "__main__":
    unittest.main()
