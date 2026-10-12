"""
Tests for Clinical Doctor Portal authentication, clinical reviews, security rules, and document management.
"""

import os
import io
import unittest
from backend.app import app


class TestDoctorPortal(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_doctor_routes(self):
        """Test that all doctor portal web routes return 200 OK"""
        routes = [
            '/doctor/login',
            '/doctor/dashboard',
            '/doctor/patient/usr_demo_ramesh',
            '/doctor/patient/usr_demo_ramesh/screening/SCR-2026-0812'
        ]
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Doctor route {route} failed with status {response.status_code}")

    def test_doctor_templates_and_styles_exist(self):
        """Test that all required doctor frontend templates and CSS files exist"""
        files = [
            'doctor-login.html',
            'doctor-dashboard.html',
            'doctor-patient.html',
            'doctor-screening.html',
            'doctor-portal.css'
        ]
        for f in files:
            file_path = os.path.join(self.root_dir, f)
            self.assertTrue(os.path.exists(file_path), f"File {f} is missing from project root")

    def test_firestore_and_storage_rules(self):
        """Verify role-based security rules for clinical doctors and reports"""
        firestore_rules_path = os.path.join(self.root_dir, 'firestore.rules')
        storage_rules_path = os.path.join(self.root_dir, 'storage.rules')

        self.assertTrue(os.path.exists(firestore_rules_path), "firestore.rules missing")
        self.assertTrue(os.path.exists(storage_rules_path), "storage.rules missing")

        with open(firestore_rules_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('match /doctors/{doctorId}', content)
            self.assertIn('match /clinicalReports/{reportId}', content)
            self.assertIn('match /versions/{versionId}', content)
            self.assertIn('function isDoctor()', content)
            self.assertIn('CLINICAL_DOCTOR', content)

        with open(storage_rules_path, 'r', encoding='utf-8') as f:
            storage_content = f.read()
            self.assertIn('match /clinical-reports/{patientUid}/{screeningId}/{reportId}/{allPaths=**}', storage_content)
            self.assertIn('CLINICAL_DOCTOR', storage_content)

    def test_firebase_config_doctor_services(self):
        """Verify DoctorAuth and DoctorClinical services in firebase-config.js"""
        config_path = os.path.join(self.root_dir, 'firebase-config.js')
        self.assertTrue(os.path.exists(config_path), "firebase-config.js missing")

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Doctor Auth
            self.assertIn('DoctorAuth', content)
            self.assertIn('DoctorClinical', content)
            self.assertIn('searchPatients', content)
            self.assertIn('getPatientProfile', content)
            self.assertIn('getScreeningDetails', content)
            self.assertIn('saveClinicalReport', content)
            self.assertIn('updateClinicalReport', content)
            self.assertIn('getReportVersions', content)
            self.assertIn('uploadClinicalDocument', content)
            self.assertIn('getDoctorReportsForScreening', content)

    def test_doctor_document_upload_api(self):
        """Test document upload REST endpoint for clinical reports"""
        test_file_content = b"%PDF-1.4 Mock Clinical Evaluation Document For Diabetic Retinopathy"
        data = {
            'file': (io.BytesIO(test_file_content), 'clinical_eval_oct.pdf'),
            'patientUid': 'usr_demo_ramesh',
            'screeningId': 'SCR-2026-0812'
        }
        response = self.client.post(
            '/api/doctor/upload-document',
            data=data,
            content_type='multipart/form-data'
        )
        self.assertIn(response.status_code, [200, 201])
        res_json = response.get_json()
        self.assertEqual(res_json.get('status'), 'success')
        doc = res_json.get('document', {})
        self.assertIn('clinical_eval_oct.pdf', doc.get('fileName'))
        self.assertIn('url', doc)
        self.assertEqual(doc.get('patientUid'), 'usr_demo_ramesh')
        self.assertEqual(doc.get('relatedScreening'), 'SCR-2026-0812')

        # Test retrieving the uploaded document
        safe_name = doc.get('safeName')
        get_res = self.client.get(f"/api/doctor/document/usr_demo_ramesh/SCR-2026-0812/{safe_name}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.data, test_file_content)


if __name__ == '__main__':
    unittest.main()
