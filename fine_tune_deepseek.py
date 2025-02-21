# Import necessary libraries
import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore
import json
import os
from datetime import datetime

# Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
DATA_PATH = os.path.join(BASE_DIR, "Data", "Raw_Data", "formatted_testdata.json")

# Configuration for experimentation
EXPERIMENT_CONFIG = {
    "learning_rate": 2e-5,
    "lora_rank": 4,
    "batch_size": 1,
    "epochs": 1,
    "gradient_accumulation_steps": 3,
    "output_dir": f"output_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "dropout": 0.1,
    "weight_decay": 0.01,
}

# Load and preprocess dataset
def load_and_preprocess_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = [{"text": item["text"]} for item in json.load(f)]
    
    # 80/10/10 split for train/validation/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    return dataset[:train_size], dataset[train_size:train_size + val_size], dataset[train_size + val_size:]

# Tokenization function
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=2048, padding="max_length")

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)  # Remove device_map and quantization

# Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=EXPERIMENT_CONFIG["lora_rank"],
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=EXPERIMENT_CONFIG["dropout"],
    bias="none",
)
model = get_peft_model(model, lora_config)

# Move model to CPU explicitly
model.to("cpu")

# Load dataset
train_data, val_data, _ = load_and_preprocess_dataset(DATA_PATH)

# Tokenize datasets
train_tokenized = [tokenize_function(item, tokenizer) for item in train_data]
val_tokenized = [tokenize_function(item, tokenizer) for item in val_data]

# Training arguments
training_args = TrainingArguments(
    output_dir=EXPERIMENT_CONFIG["output_dir"],
    learning_rate=EXPERIMENT_CONFIG["learning_rate"],
    per_device_train_batch_size=EXPERIMENT_CONFIG["batch_size"],
    per_device_eval_batch_size=EXPERIMENT_CONFIG["batch_size"],
    num_train_epochs=EXPERIMENT_CONFIG["epochs"],
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=False,  # Disable mixed precision (not supported on CPU)
    gradient_accumulation_steps=EXPERIMENT_CONFIG["gradient_accumulation_steps"],
    push_to_hub=False,
    weight_decay=EXPERIMENT_CONFIG["weight_decay"],
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
)

# Start training
print(f"Starting experiment with config: {EXPERIMENT_CONFIG}")
trainer.train()

# Save evaluation results
eval_results = trainer.evaluate()
print(f"Evaluation Results (KPIs): {eval_results}")

# Save model
model.save_pretrained(os.path.join(EXPERIMENT_CONFIG["output_dir"], "final_model"))
tokenizer.save_pretrained(os.path.join(EXPERIMENT_CONFIG["output_dir"], "final_tokenizer"))

print(f"Experiment completed! Check {EXPERIMENT_CONFIG['output_dir']} for results and KPIs.")