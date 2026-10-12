# AI DR Screen Chatbot — Testing & Evaluation Guide

## 1. Test Suite Architecture
The test suite consists of two complementary layers:
1. **Unit & Integration Tests (`tests/test_chatbot.py`)**: Tests safety regexes, emergency triage, session context management, referral rules, RAG retriever, and Flask REST routes.
2. **Benchmark Evaluation Dataset (`tests/test_chatbot_eval.py` & `tests/chatbot_test_cases.json`)**: Contains 120 structured evaluation questions across 11 key domains.

## 2. Running Automated Tests

### Running Chatbot Unit & Integration Tests
```powershell
.\.venv\Scripts\python -m unittest tests/test_chatbot.py
```

### Running the 120-Case Evaluation Dataset
```powershell
.\.venv\Scripts\python -m unittest tests/test_chatbot_eval.py
```

### Running the Entire Application Test Suite (Zero Regressions)
```powershell
.\.venv\Scripts\python -m unittest discover tests/
```

## 3. Evaluation Dataset Categories
The dataset in `tests/chatbot_test_cases.json` includes:
- 20 General questions (overview, device support, browsers)
- 15 Image upload questions (formats, sizes, OD/OS)
- 10 Image quality assessment (IQA) questions
- 10 DR screening & EfficientNet questions
- 10 Explainability questions (Grad-CAM, SHAP, U-Net)
- 10 Medical report & OCR questions
- 10 Risk and deterministic referral questions
- 5 Website navigation & keyboard shortcut questions
- 5 Verified healthcare facility questions
- 5 Privacy & data security questions
- 20 Medical safety, emergency symptom, and adversarial injection questions
