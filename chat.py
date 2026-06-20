from transformers import AutoModelForCausalLM, AutoTokenizer


def chat():
    model = AutoModelForCausalLM.from_pretrained("output/best_model")
    tokenizer = AutoTokenizer.from_pretrained("output/best_model")
    model = model.eval()

    print("Tiny LoRA Chat (type /exit to quit)\n")
    try:
        while True:
            user_input = input("You: ")
            if user_input in ("/exit", "/quit"):
                break

        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")

        with __import__("torch").no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"Bot: {response}\n")
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    chat()
