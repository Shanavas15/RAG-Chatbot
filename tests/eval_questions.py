import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import EnterpriseRAGPipeline

EVAL_SET = [
    {"question": "What loss function does Logistic Regression minimize?", "expect_keyword": "cross-entropy"},
    {"question": "What is the difference between bagging and boosting?", "expect_keyword": "variance"},
    {"question": "Why does KNN require feature scaling?", "expect_keyword": "distance"},
    {"question": "What problem does the vanishing gradient cause in RNNs?", "expect_keyword": "long-term"},
    {"question": "What is the formula for F1 score?", "expect_keyword": "harmonic"},
    {"question": "How does XGBoost handle regularization?", "expect_keyword": "L1"},
    {"question": "What does the elbow method help choose in K-Means?", "expect_keyword": "k"},
    {"question": "What is the kernel trick used for in SVM?", "expect_keyword": "non-linear"},
    {"question": "What does AUC of 0.5 mean in ROC-AUC?", "expect_keyword": "random"},
    {"question": "What is the purpose of the forget gate in LSTM?", "expect_keyword": "discard"},
]

def run_eval():
    pipeline = EnterpriseRAGPipeline(vector_store_path="vector_store")
    passed, failed = 0, 0

    for case in EVAL_SET:
        answer, sources, is_web = pipeline.ask(case["question"], top_k=3)
        keyword_hit = case["expect_keyword"].lower() in answer.lower()
        status = "PASS" if keyword_hit and not is_web else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"[{status}] Q: {case['question']}")
        print(f"   web_fallback={is_web} answer_snippet={answer[:120]!r}")

    print(f"\n{passed} passed, {failed} failed out of {len(EVAL_SET)}")

if __name__ == "__main__":
    run_eval()