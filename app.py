import os
from huggingface_hub import InferenceClient

# --- SETUP ---
# Get your free token from: https://huggingface.co/settings/tokens
HF_TOKEN = "your_huggingface_token_here"

# Model choice: Gemma-2-2b-it is fast and lightweight for free-tier inference
MODEL_ID = "google/gemma-2-2b-it"

client = InferenceClient(api_key=HF_TOKEN)

def get_health_advice(user_input, history):
    system_prompt = (
        "You are a professional Health and Diet Assistant. "
        "Provide concise, evidence-based advice on nutrition, exercise, and wellness. "
        "Always include a disclaimer that you are an AI and not a doctor."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history for context
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    messages.append({"role": "user", "content": user_input})

    try:
        # Use chat_completion for a natural conversational flow
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🌿 NutriBot: Your AI Health & Diet Guide (Type 'quit' to exit)")
    chat_history = []
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["quit", "exit", "bye"]:
            print("Stay healthy! Goodbye.")
            break
            
        advice = get_health_advice(user_query, chat_history)
        print(f"\nNutriBot: {advice}")
        
        # Keep history to 5 turns to manage token limits on free tier
        chat_history.append((user_query, advice))
        if len(chat_history) > 5:
            chat_history.pop(0)

if __name__ == "__main__":
    main()