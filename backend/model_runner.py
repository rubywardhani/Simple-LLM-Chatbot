from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading DialoGPT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
print("Model loaded successfully!")

chat_history_ids = None


def generate_response(user_input: str) -> str:
    """
    Generate response dari user input menggunakan DialoGPT
    """
    global chat_history_ids

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids],
                              dim=-1) if chat_history_ids is not None else new_input_ids

    if bot_input_ids.shape[-1] > 1000:
        bot_input_ids = bot_input_ids[:, -1000:]

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return response.strip()


def reset_chat_history():
    """Reset chat history untuk mulai percakapan baru"""
    global chat_history_ids
    chat_history_ids = None
    print("Chat history has been reset")
