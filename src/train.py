# src/train.py

import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, OUTPUT_DIR, TRAIN_FILE

# 1. Load and prepare dataset
df = pd.read_csv(TRAIN_FILE)[["nl_input", "sql_query"]]
dataset = Dataset.from_pandas(df)

# 2. Tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    inputs = tokenizer(batch["nl_input"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    targets = tokenizer(batch["sql_query"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Load model
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 4. Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir="logs",
    logging_steps=1,
    save_total_limit=1,
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 6. Train
if __name__ == "__main__":
    trainer.train()
