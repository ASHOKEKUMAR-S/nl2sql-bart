import os
import pandas as pd
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from evaluate import load as load_metric
import textdistance

from src.db_utils import create_db_engine, execute_sql
from src.config import MODEL_NAME, MAX_LENGTH

# Paths
MODEL_DIR = "models/bart-nl2sql/final"
EVAL_FILE = "data/eval.csv"
OUTPUT_CSV = "outputs/eval_summary.csv"

# Load model and tokenizer
print("🔄 Loading model and tokenizer...")
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Load metrics
bleu_metric = load_metric("bleu")

# Load evaluation data
print("📄 Reading evaluation file...")
df = pd.read_csv(EVAL_FILE)[["nl_input", "sql_query"]]

# MySQL DB connection
engine = create_db_engine()

# Store results
eval_results = []

# Helper: Generate SQL from natural language
def generate_sql(nl_input: str) -> str:
    inputs = tokenizer(nl_input, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    outputs = model.generate(inputs["input_ids"], max_length=MAX_LENGTH, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluation loop
print("🚀 Starting evaluation...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    nl = row["nl_input"]
    expected_sql = row["sql_query"]

    predicted_sql = generate_sql(nl)

    # Execution check
    expected_result = execute_sql(expected_sql, engine)
    predicted_result = execute_sql(predicted_sql, engine)
    execution_match = (
        isinstance(expected_result, pd.DataFrame) and
        isinstance(predicted_result, pd.DataFrame) and
        expected_result.equals(predicted_result)
    )

    # Metrics
    exact_match = predicted_sql.strip().lower() == expected_sql.strip().lower()
    bleu = bleu_metric.compute(
        predictions=[predicted_sql],
        references=[[expected_sql]]
    )["bleu"]
    levenshtein_sim = textdistance.levenshtein.normalized_similarity(predicted_sql, expected_sql)

    # Log
    eval_results.append({
        "nl_input": nl,
        "expected_sql": expected_sql,
        "predicted_sql": predicted_sql,
        "exact_match": exact_match,
        "execution_match": execution_match,
        "bleu_score": round(bleu, 4),
        "levenshtein_similarity": round(levenshtein_sim, 4),
        "error_expected": None if isinstance(expected_result, pd.DataFrame) else expected_result,
        "error_predicted": None if isinstance(predicted_result, pd.DataFrame) else predicted_result
    })

# Save evaluation log
os.makedirs("outputs", exist_ok=True)
pd.DataFrame(eval_results).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Evaluation complete. Summary saved to: {OUTPUT_CSV}")

# Summary stats
total = len(eval_results)
em = sum(r["exact_match"] for r in eval_results)
ex = sum(r["execution_match"] for r in eval_results)

print(f"\n📊 Summary:")
print(f" - Total samples         : {total}")
print(f" - Exact Match count     : {em} ({em / total:.2%})")
print(f" - Execution Match count : {ex} ({ex / total:.2%})")
