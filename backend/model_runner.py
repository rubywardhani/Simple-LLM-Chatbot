from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading rbGPT model...")
tokenizer = AutoTokenizer.from_pretrained("rubywardhani/rbGPT")
model = AutoModelForCausalLM.from_pretrained("rubywardhani/rbGPT")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

chat_history_ids = None


def generate_response(user_input: str) -> str:
    """
    Generate response from user input using rbGPT
    """
    global chat_history_ids

    formatted_input = f"User: {user_input}\nBot:"

    # Encode input with attention mask
    inputs = tokenizer.encode_plus(
        formatted_input,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if chat_history_ids is not None:
        if chat_history_ids.shape[-1] > 800:
            chat_history_ids = chat_history_ids[:, -800:]

        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        attention_mask = torch.ones_like(input_ids)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[-1] + 50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2
        )

    # Update chat history
    chat_history_ids = output

    response = tokenizer.decode(
        output[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return response.strip()


def reset_chat_history():
    """Reset chat history to start a new conversation"""
    global chat_history_ids
    chat_history_ids = None
    print("Chat history has been reset")
