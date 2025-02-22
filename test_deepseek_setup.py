import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration
MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
DATA_FILE = "prepared_deepseek_training_data.jsonl"
CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}<｜User｜>{{ message['content'] }}{% elif message['role'] == 'assistant' %}<｜Assistant｜>{{ message['content'] }}{% endif %}{% endfor %}"""

def test_tokenizer_and_template():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.add_special_tokens({"additional_special_tokens": ["<｜User｜>", "<｜Assistant｜>"]})
    
    # Load a sample from data
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
    
    # Apply chat template (no tokenization)
    formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    print("=== FORMATTED TEXT ===")
    print(formatted)
    
    # Apply chat template with tokenization
    encoded = tokenizer.apply_chat_template(sample["messages"], tokenize=True, return_tensors="pt")
    
    # Access input_ids directly and print shape
    input_ids = encoded["input_ids"]  # 2D tensor [1, seq_len]
    print(f"\n=== TENSOR SHAPE: {input_ids.shape} ===")
    
    # Get token count (sequence length, assuming batch size is 1)
    token_count = input_ids.shape[1]
    print(f"=== TOKEN COUNT: {token_count} ===")
    
    # Decode the tokenized input
    decoded = tokenizer.decode(input_ids[0])  # Index 0 to get the first (and only) sequence
    print("\n=== DECODED TEXT ===")
    print(decoded)
    
    # Test role detection
    print("\n=== ROLE DETECTION ===")
    user_token = tokenizer.encode("<｜User｜>", add_special_tokens=False)[0]
    assistant_token = tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)[0]
    
    print(f"User token ID: {user_token}")
    print(f"Assistant token ID: {assistant_token}")
    
    # Create labels (mask user sections with -100)
    labels = input_ids.clone()  # Clone the 2D tensor [1, seq_len]
    sequence = input_ids[0]     # Get the 1D sequence for iteration
    
    is_user_section = False
    label_mask = torch.ones_like(sequence)  # 1D mask for the sequence
    
    for j, token_id in enumerate(sequence):
        if token_id == user_token:
            is_user_section = True
            print(f"Position {j}: User section starts")
        elif token_id == assistant_token:
            is_user_section = False
            print(f"Position {j}: Assistant section starts")
        
        if is_user_section:
            label_mask[j] = 0
    
    # Apply mask to labels (-100 for user sections)
    labels[0] = torch.where(label_mask == 1, sequence, -100)  # Apply to the sequence
    
    # Check label distribution
    total_tokens = labels.numel()
    masked_tokens = (labels == -100).sum().item()
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Masked tokens (user sections): {masked_tokens} ({masked_tokens/total_tokens:.2%})")
    print(f"Training tokens (assistant sections): {total_tokens - masked_tokens} ({(total_tokens - masked_tokens)/total_tokens:.2%})")
    
    return True

def test_forward_pass():
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.chat_template = CHAT_TEMPLATE
        tokenizer.add_special_tokens({"additional_special_tokens": ["<｜User｜>", "<｜Assistant｜>"]})
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"  # Avoid sdpa warning
        )
        model.resize_token_embeddings(len(tokenizer))
        
        # Load sample data
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
        
        # Prepare inputs with labels
        inputs = tokenizer.apply_chat_template(sample["messages"], tokenize=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        # Create proper labels
        labels = input_ids.clone()
        user_token = tokenizer.encode("<｜User｜>", add_special_tokens=False)[0]
        assistant_token = tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)[0]
        
        is_user_section = False
        label_mask = torch.ones_like(input_ids[0])  # 1D mask for the sequence
        
        for j, token_id in enumerate(input_ids[0]):
            if token_id == user_token:
                is_user_section = True
            elif token_id == assistant_token:
                is_user_section = False
            
            if is_user_section:
                label_mask[j] = 0
        
        # Apply mask to labels (-100 for user sections)
        labels[0] = torch.where(label_mask == 1, input_ids[0], -100)
        labels = labels.to(model.device)
        
        # Forward pass with labels
        print("\n=== TESTING FORWARD PASS WITH LABELS ===")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # Check if we got loss and logits
        print(f"Loss computed: {outputs.loss.item()}")
        print(f"Output keys: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Error in forward pass test: {e}")
        return False

if __name__ == "__main__":
    print("=== TESTING TOKENIZER AND CHAT TEMPLATE ===")
    tokenizer_test_passed = test_tokenizer_and_template()
    
    if tokenizer_test_passed:
        print("\n=== TOKENIZER TEST PASSED ===")
        
        print("\n=== TESTING MODEL FORWARD PASS ===")
        forward_test_passed = test_forward_pass()
        
        if forward_test_passed:
            print("\n✅ ALL TESTS PASSED - YOUR SETUP SHOULD WORK FOR TRAINING")
        else:
            print("\n❌ FORWARD PASS TEST FAILED - CHECK ERROR MESSAGES")
    else:
        print("\n❌ TOKENIZER TEST FAILED")