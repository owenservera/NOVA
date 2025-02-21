from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download and load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer locally (optional)
model.save_pretrained("./deepseek-model")
tokenizer.save_pretrained("./deepseek-tokenizer")

print("Model and tokenizer downloaded and saved successfully!")