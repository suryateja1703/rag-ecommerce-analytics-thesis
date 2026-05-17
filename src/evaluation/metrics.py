import os
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from dotenv import load_dotenv
import mlflow
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def compute_rouge(predictions, references):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    print("\n📊 Computing ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    results = {
        "rouge1": round(sum(r1)/len(r1), 4),
        "rouge2": round(sum(r2)/len(r2), 4),
        "rougeL": round(sum(rl)/len(rl), 4),
    }
    print(f"✅ ROUGE-1: {results['rouge1']} | ROUGE-2: {results['rouge2']} | ROUGE-L: {results['rougeL']}")
    return results

def compute_bertscore(predictions, references):
    """Compute BERTScore F1."""
    print("\n📊 Computing BERTScore...")
    P, R, F1 = bert_score(
        predictions, references,
        lang="en", verbose=False
    )
    results = {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall"   : round(R.mean().item(), 4),
        "bertscore_f1"       : round(F1.mean().item(), 4),
    }
    print(f"✅ BERTScore F1: {results['bertscore_f1']}")
    return results

def compute_faithfulness(predictions, contexts):
    """Measure how grounded answers are in retrieved context."""
    print("\n📊 Computing Faithfulness...")
    scores = []
    for pred, ctx in zip(predictions, contexts):
        pred_words = set(pred.lower().split())
        ctx_words  = set(ctx.lower().split())
        overlap    = len(pred_words & ctx_words) / len(pred_words) if pred_words else 0
        scores.append(overlap)
    result = {"faithfulness": round(sum(scores)/len(scores), 4)}
    print(f"✅ Faithfulness: {result['faithfulness']}")
    return result

def compute_answer_relevance(queries, predictions):
    """Simple relevance check — keyword overlap."""
    print("\n📊 Computing Answer Relevance...")
    scores = []
    for query, pred in zip(queries, predictions):
        query_words = set(query.lower().split())
        pred_words  = set(pred.lower().split())
        overlap     = len(query_words & pred_words) / len(query_words) if query_words else 0
        scores.append(overlap)
    result = {"answer_relevance": round(sum(scores)/len(scores), 4)}
    print(f"✅ Answer Relevance: {result['answer_relevance']}")
    return result

def run_evaluation(predictions, references, contexts, queries, run_name="evaluation"):
    """Run full evaluation suite and log to MLflow."""
    print("=" * 60)
    print(f"🚀 Running Evaluation: {run_name}")
    print(f"📊 Evaluating {len(predictions)} predictions")
    print("=" * 60)

    with mlflow.start_run(run_name=run_name):
        # Compute all metrics
        rouge     = compute_rouge(predictions, references)
        bert      = compute_bertscore(predictions, references)
        faith     = compute_faithfulness(predictions, contexts)
        relevance = compute_answer_relevance(queries, predictions)

        # Combine all metrics
        all_metrics = {**rouge, **bert, **faith, **relevance}

        # Log to MLflow
        for metric, value in all_metrics.items():
            mlflow.log_metric(metric, value)
        mlflow.log_param("num_predictions", len(predictions))
        mlflow.log_param("model", os.getenv("PRIMARY_MODEL", "claude-opus-4-6"))

        # Save results CSV
        results_df = pd.DataFrame([all_metrics])
        results_df.to_csv("evaluation_results.csv", index=False)
        mlflow.log_artifact("evaluation_results.csv")

        print("\n" + "=" * 60)
        print("✅ Evaluation Complete!")
        print("📊 Results:")
        for k, v in all_metrics.items():
            print(f"   {k:25s}: {v}")
        print("=" * 60)

    return all_metrics

if __name__ == "__main__":
    # Test with sample data
    preds = [
        "The Sony WH-1000XM5 offers excellent noise cancellation and 30 hour battery life making it best under $300.",
        "Apple AirPods Pro features active noise cancellation and spatial audio with seamless iPhone integration."
    ]
    refs = [
        "Sony headphones provide strong noise cancellation with long battery performance.",
        "Apple AirPods Pro offers noise cancellation and works well with Apple devices."
    ]
    ctxs = [
        "Sony WH-1000XM5 features industry leading noise cancellation and 30 hours battery life priced at $279.",
        "Apple AirPods Pro has active noise cancellation spatial audio and transparency mode for $249."
    ]
    queries = [
        "What are the best noise cancelling headphones?",
        "Which wireless earbuds work best with iPhone?"
    ]

    mlflow.set_experiment("evaluation_test")
    run_evaluation(preds, refs, ctxs, queries, run_name="test_run")