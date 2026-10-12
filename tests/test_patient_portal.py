"""
Tests for Patient Portal authentication, routing, security rules, and screening integration.
"""

import os
import unittest
from backend.app import app


class TestPatientPortal(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_patient_routes(self):
        routes = [
            '/patient/login',
            '/patient/signup',
            '/patient/dashboard',
            '/patient/screening',
            '/patient/profile',
            '/patient/screening/SCR-TEST1-123'
        ]
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with status {response.status_code}")

    def test_security_rules_exist(self):
        firestore_rules_path = os.path.join(self.root_dir, 'firestore.rules')
        storage_rules_path = os.path.join(self.root_dir, 'storage.rules')

        self.assertTrue(os.path.exists(firestore_rules_path), "firestore.rules missing")
        self.assertTrue(os.path.exists(storage_rules_path), "storage.rules missing")

        with open(firestore_rules_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('match /patients/{userId}', content)
            self.assertIn('request.auth.uid == userId', content)
            self.assertIn('match /screenings/{screeningId}', content)
            self.assertIn('resource.data.patientUid == request.auth.uid', content)

    def test_firebase_config_methods(self):
        config_path = os.path.join(self.root_dir, 'firebase-config.js')
        self.assertTrue(os.path.exists(config_path), "firebase-config.js missing")

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('signUp(', content)
            self.assertIn('login(', content)
            self.assertIn('saveScreening(', content)
            self.assertIn('getPatientScreenings(', content)
            self.assertIn('getScreeningById(', content)
            self.assertIn('updateScreeningExplanations(', content)

    def test_patient_portal_templates_exist(self):
        pages = [
            'patient-login.html',
            'patient-signup.html',
            'patient-dashboard.html',
            'patient-screening.html',
            'patient-profile.html',
            'patient-portal.css'
        ]
        for p in pages:
            path = os.path.join(self.root_dir, p)
            self.assertTrue(os.path.exists(path), f"File {p} does not exist")


if __name__ == '__main__':
    unittest.main()
