"""
Automated unit tests verifying unified Staff & Doctor login,
absence of separate doctor login banner, and Doctor profile & patients on dashboard.
"""

import os
import unittest
from backend.app import app


class TestStaffDoctorUnifiedLogin(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_login_page_unified_and_no_separate_doctor_banner(self):
        """Verify that login.html does NOT contain the separate doctor portal banner and provides unified staff & doctor login"""
        login_path = os.path.join(self.root_dir, 'login.html')
        self.assertTrue(os.path.exists(login_path))
        with open(login_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Must NOT contain the separate doctor banner linking to doctor-login.html
        self.assertNotIn('Clinical Doctor & Specialist Portal', content)
        self.assertNotIn('doctor-login.html', content)

        # Must contain unified Staff & Doctor login
        self.assertIn('Staff &amp; Doctor Sign In', content)
        self.assertIn('Staff &amp; Doctor Login', content)
        self.assertIn('Doctor / Specialist', content)
        self.assertIn('Health Center Staff / PHC', content)
        self.assertIn('Dr. Anitha (Doctor)', content)
        self.assertIn('Priya (Health Center Staff)', content)
        self.assertIn('id="btn-login-label"', content)

    def test_doctor_login_redirects_to_unified_login(self):
        """Verify that doctor-login.html auto-redirects to unified login.html?role=doctor"""
        doc_login_path = os.path.join(self.root_dir, 'doctor-login.html')
        self.assertTrue(os.path.exists(doc_login_path))
        with open(doc_login_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn("window.location.replace('login.html?role=doctor')", content)

    def test_doctor_dashboard_has_profile_and_patients(self):
        """Verify that doctor-dashboard.html displays the Doctor Profile Card and Assigned Patients Registry"""
        doc_dash_path = os.path.join(self.root_dir, 'doctor-dashboard.html')
        self.assertTrue(os.path.exists(doc_dash_path))
        with open(doc_dash_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Doctor Profile Card
        self.assertIn('id="profile-section"', content)
        self.assertIn('id="profile-doc-fullname"', content)
        self.assertIn('id="profile-meta-mci"', content)
        self.assertIn('id="profile-meta-abdm"', content)
        self.assertIn('id="profile-meta-facility"', content)
        self.assertIn('id="profile-meta-camps"', content)
        self.assertIn('id="profile-edit-modal"', content)

        # Assigned Patients Registry
        self.assertIn('id="patients-section"', content)
        self.assertIn("My Assigned Patients &amp; Clinical Registry", content)
        self.assertIn('id="patient-search-input"', content)
        self.assertIn('id="patient-table-body"', content)
        self.assertIn('applyPatientFilter', content)

    def test_login_js_routes_doctor_to_dashboard(self):
        """Verify that login.js sets doctor session and redirects to doctor-dashboard.html"""
        login_js_path = os.path.join(self.root_dir, 'login.js')
        self.assertTrue(os.path.exists(login_js_path))
        with open(login_js_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('aidr_doctor_auth_user', content)
        self.assertIn("window.location.href = 'doctor-dashboard.html'", content)
        self.assertIn("params.get('role')", content)


if __name__ == '__main__':
    unittest.main()
