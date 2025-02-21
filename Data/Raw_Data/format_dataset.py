import json

def format_conversation(example):
    """
    Format a conversation into a prompt-response pair, handling the specific structure of your dataset.
    """
    prompt_parts = []
    response_parts = []

    # Safely iterate through messages, checking for content and data
    for msg in example.get("messages", []):
        role = msg.get("role", "unknown")
        content = msg.get("content", [])

        # Handle the content structure (list of dictionaries with 'type' and 'data')
        data = ""
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "data" in content[0]:
            data = content[0]["data"]
        elif isinstance(content, str):  # Fallback for unexpected string content
            data = content
        else:
            data = ""  # Default to empty if we can't find data

        if role == "user":
            prompt_parts.append(f"{role}: {data}")
        elif role == "assistant":
            response_parts.append(f"{role}: {data}")

    # Join the parts with newlines, handling empty cases
    prompt = "\n".join(prompt_parts) if prompt_parts else "User: No prompt available"
    response = "\n".join(response_parts) if response_parts else "Assistant: No response available"

    return {"text": f"{prompt}\nAssistant: {response}"}

# Load and process the dataset
with open("cleaned_testdata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Format the conversations
formatted_data = [format_conversation(entry) for entry in data]

# Save the formatted data
with open("formatted_testdata.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2)

print("Dataset formatted successfully! Check formatted_testdata.json for the results.")
