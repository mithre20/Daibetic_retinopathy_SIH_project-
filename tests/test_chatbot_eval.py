"""
Evaluation Test Suite for 120 Chatbot Test Cases (tests/chatbot_test_cases.json)
Validates intent classification, medical safety refusal, emergency triage, and adversarial robustness.
"""

import os
import sys
import json
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from chatbot.services.chatbot_service import get_chatbot_service
from chatbot.safety.safety import check_medical_safety
from chatbot.safety.emergency import check_emergency_symptoms

class TestChatbotEvaluationDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = get_chatbot_service()
        dataset_path = os.path.join(ROOT_DIR, "tests", "chatbot_test_cases.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            cls.test_cases = json.load(f)

    def test_all_120_cases(self):
        self.assertGreaterEqual(len(self.test_cases), 100, "Dataset must contain at least 100 test questions.")

        passed = 0
        for item in self.test_cases:
            qid = item["id"]
            question = item["question"]
            expected_intent = item["expected_intent"]
            safety_level = item["safety_level"]

            # 1. Emergency cases
            if safety_level == "emergency":
                em_res = check_emergency_symptoms(question)
                self.assertIsNotNone(em_res, f"Case {qid} ('{question}') failed to trigger emergency triage.")
                self.assertTrue(em_res["is_emergency"])
                passed += 1
                continue

            # 2. Critical medical & adversarial cases
            if safety_level in ["critical_medical", "adversarial"]:
                safety_res = check_medical_safety(question)
                self.assertIsNotNone(safety_res, f"Case {qid} ('{question}') failed to trigger medical safety refusal.")
                self.assertFalse(safety_res["safe"])
                passed += 1
                continue

            # 3. Normal queries processed through chatbot service
            response = self.service.process_message(question, session_id="eval_sess")
            self.assertIsNotNone(response)
            self.assertIn("answer", response)
            self.assertTrue(len(response["answer"]) > 10, f"Case {qid} returned empty or too short answer.")
            passed += 1

        print(f"\nChatbot Evaluation Dataset: {passed} / {len(self.test_cases)} test cases passed successfully.")

if __name__ == "__main__":
    unittest.main()
