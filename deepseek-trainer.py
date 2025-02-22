import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
DATA_FILE = "prepared_deepseek_training_data.jsonl"

def load_and_prepare_data():
    logger.info(f"Loading dataset from {DATA_FILE}")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset

def tokenize_function(examples, tokenizer):
    formatted_text = examples["formatted_text"]
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=3000,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].squeeze(0)
    attention_mask = tokenized["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    user_token = tokenizer.encode("<｜User｜>", add_special_tokens=False)[0]
    assistant_token = tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    is_user_section = False
    for i, token_id in enumerate(input_ids):
        if token_id == user_token:
            is_user_section = True
        elif token_id == assistant_token:
            is_user_section = False
        if is_user_section:
            labels[i] = -100
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist()
    }

def train():
    logger.info(f"Loading tokenizer for {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = {"additional_special_tokens": ["<｜User｜>", "<｜Assistant｜>"]}
    tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {len(special_tokens['additional_special_tokens'])} special tokens to tokenizer")

    train_dataset, eval_dataset = load_and_prepare_data()

    logger.info(f"Loading model {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Changed for CPU stability
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=train_dataset.column_names,
        num_proc=2  # Multi-threaded data processing
    )
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=eval_dataset.column_names,
        num_proc=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=1,  # Reduced batch size for CPU
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,  # More frequent logs
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=False,  # Disabled for CPU stability
        fp16=False,
        use_cache=False  # Disabled cache to save memory
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        verbose=True  # More detailed logs
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    train()
