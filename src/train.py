import os
import time
import pandas as pd
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer,
    IntervalStrategy
)

from src.config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    TRAIN_FILE
)

# Step 1: Load data
df = pd.read_csv(TRAIN_FILE)[["nl_input", "sql_query"]]
dataset = Dataset.from_pandas(df)


# Step 2: Tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    inputs = tokenizer(batch["nl_input"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    targets = tokenizer(batch["sql_query"], max_length=MAX_LENGTH, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 3: Load model
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Step 4: Custom Trainer with epoch time logging
class LoggingTrainer(Trainer):
    def train(self, *args, **kwargs):
        start = time.time()
        print(f"🚀 Training started at: {time.ctime(start)}")
        result = super().train(*args, **kwargs)
        end = time.time()
        print(f"✅ Training completed at: {time.ctime(end)}")
        print(f"⏱️ Total training time: {end - start:.2f} seconds")
        return result

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir="logs",
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=1,
    logging_first_step=True,
    save_strategy=IntervalStrategy.EPOCH,
    save_total_limit=1,
    report_to="none",  # set to "tensorboard" if you want tb logs
    disable_tqdm=False
)

# Step 6: Trainer
trainer = LoggingTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Step 7: Start training
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training finished!")
