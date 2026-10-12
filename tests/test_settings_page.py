import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app

class TestSettingsPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_settings_files_exist(self):
        """Verify all settings frontend files exist and are well-formed."""
        settings_html = os.path.join(ROOT_DIR, "settings.html")
        settings_css = os.path.join(ROOT_DIR, "settings.css")
        settings_js = os.path.join(ROOT_DIR, "settings.js")

        self.assertTrue(os.path.exists(settings_html), "settings.html must exist")
        self.assertTrue(os.path.exists(settings_css), "settings.css must exist")
        self.assertTrue(os.path.exists(settings_js), "settings.js must exist")

        self.assertGreater(os.path.getsize(settings_html), 5000, "settings.html should be complete")
        self.assertGreater(os.path.getsize(settings_css), 5000, "settings.css should contain full design system")
        self.assertGreater(os.path.getsize(settings_js), 3000, "settings.js should contain logic controller")

    def test_settings_route_http(self):
        """Verify /settings and /settings.html both return HTTP 200 with HTML content."""
        res = self.client.get("/settings")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"System Settings &amp; Preferences", res.data)
        self.assertIn(b"AI DR Screen", res.data)

        res2 = self.client.get("/settings.html")
        self.assertEqual(res2.status_code, 200)

    def test_all_12_settings_categories_present(self):
        """Verify all 12 requested categories are defined in settings.html."""
        settings_html_path = os.path.join(ROOT_DIR, "settings.html")
        with open(settings_html_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_panels = [
            'id="panel-general"',
            'id="panel-appearance"',
            'id="panel-notifications"',
            'id="panel-screening"',
            'id="panel-ai"',
            'id="panel-reports"',
            'id="panel-privacy"',
            'id="panel-profile"',
            'id="panel-language"',
            'id="panel-device"',
            'id="panel-sync"',
            'id="panel-about"'
        ]

        for panel in expected_panels:
            self.assertIn(panel, content, f"Expected category panel {panel} to exist in settings.html")

    def test_ai_models_and_explainability_info(self):
        """Verify the AI & Explainability section specifies the core models and XAI techniques."""
        settings_html_path = os.path.join(ROOT_DIR, "settings.html")
        with open(settings_html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("EfficientNet-B0", content)
        self.assertIn("ResNet50", content)
        self.assertIn("U-Net", content)
        self.assertIn("Grad-CAM", content)
        self.assertIn("SHAP", content)

    def test_navigation_and_sidebar_consistency(self):
        """Verify sidebar contains the active Settings link and points back to dashboard and portals."""
        settings_html_path = os.path.join(ROOT_DIR, "settings.html")
        with open(settings_html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('class="nav-link active" data-page="settings"', content)
        self.assertIn('href="dashboard.html"', content)

    def test_settings_cross_navigation(self):
        """Verify settings is accessible from portal shell, dashboard, medivision, and doctor portal."""
        index_html = os.path.join(ROOT_DIR, "index.html")
        portal_js = os.path.join(ROOT_DIR, "portal.js")
        dashboard_html = os.path.join(ROOT_DIR, "dashboard.html")
        doctor_dashboard_html = os.path.join(ROOT_DIR, "doctor-dashboard.html")
        medivision_html = os.path.join(ROOT_DIR, "medivision.html")

        with open(index_html, "r", encoding="utf-8") as f:
            idx = f.read()
        self.assertIn("portal-tab-settings", idx)
        self.assertIn("portal-panel-settings", idx)

        with open(portal_js, "r", encoding="utf-8") as f:
            pjs = f.read()
        self.assertIn("'settings'", pjs)

        with open(dashboard_html, "r", encoding="utf-8") as f:
            dash = f.read()
        self.assertIn("settings.html", dash)

        with open(doctor_dashboard_html, "r", encoding="utf-8") as f:
            doc = f.read()
        self.assertIn("settings.html", doc)

        with open(medivision_html, "r", encoding="utf-8") as f:
            medi = f.read()
        self.assertIn("settings.html", medi)

if __name__ == "__main__":
    unittest.main()
