import json
import requests
import re
from collections import Counter

with open("eval_questions.json") as f:
    tests = json.load(f)

PASS = 0
FAIL = 0
TOTAL = len(tests)

print("\n============================")
print(" MULTI-MODAL RAG EVALUATION")
print("============================\n")

def extract_numbers(text):
    """Extract all numeric values (ints + floats) from text"""
    return re.findall(r"\d+\.?\d*", text)

def normalize_number(n):
    """Normalize numeric strings"""
    try:
        return round(float(n), 3)
    except:
        return None

for i, test in enumerate(tests, 1):
    print(f"Test {i}: {test['question']}")

    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"question": test["question"]}
    ).json()

    answer = response["answer"]
    citations = response["citations"]

    score = 0
    max_score = 3

    answer_lower = answer.lower()
    found_numbers = [normalize_number(n) for n in extract_numbers(answer)]
    found_numbers = [n for n in found_numbers if n is not None]

    # --------------------------------------------------
    # 1️⃣ Content check (number or keyword)
    # --------------------------------------------------
    content_ok = False
    for term in test["must_contain"]:
        try:
            target = normalize_number(term)
            if target in found_numbers:
                content_ok = True
        except:
            if term.lower() in answer_lower:
                content_ok = True

    if content_ok:
        score += 1

    # --------------------------------------------------
    # 2️⃣ Page citation check
    # --------------------------------------------------
    citation_ok = False
    for page in test["expected_pages"]:
        if f"page {page}" in citations.lower():
            citation_ok = True

    if citation_ok:
        score += 1

    # --------------------------------------------------
    # 3️⃣ Modality check (table / text / image)
    # --------------------------------------------------
    if "expected_modality" in test:
        if test["expected_modality"].lower() in citations.lower():
            score += 1
    else:
        # If not specified, assume OK
        score += 1

    # --------------------------------------------------
    # Final verdict
    # --------------------------------------------------
    if score == max_score:
        print("   PASS")
        PASS += 1
    else:
        print(f"   FAIL ({score}/{max_score})")
        print("   Answer:", answer)
        print("   Citations:", citations)
        FAIL += 1

print("\n============================")
print(" FINAL SCORE")
print("============================")
print(f"TOTAL: {TOTAL}")
print(f"PASS:  {PASS}")
print(f"FAIL:  {FAIL}")
print(f"Accuracy: {round(PASS/TOTAL*100,2)}%")