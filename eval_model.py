import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your fine-tuned model directory
model_save_path = "./fine-tuned-la-mini-gpt-1.5b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelForCausalLM.from_pretrained(model_save_path)

# If no pad token is set, set it explicitly to avoid warnings.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Construct a more detailed and domain-specific prompt.
prompt = (
    "You are an expert tennis coach specializing in doubles partnership compatibility. "
    "Your task is to analyze a player's statistics and match performance to determine the best compatible doubles partner. "
    "Only answer with the names of the compatible partners (e.g., 'Player 2, Player 4') and do not include any additional commentary.\n\n"
    "Player Data:\n"
    "Player ID: 5\n"
    "Ad Side Performance: 0.88\n"
    "Deuce Side Performance: 0.79\n"
    "Rating: 3.2\n"
    "Match Details: Won 6-3, 7-5\n\n"
    "Based solely on the above data, provide the best doubles partner(s) for this player. "
    "Answer ONLY with the partner names and end your response with the marker '###'.\n"
)

# Tokenize the prompt and include the attention mask.
encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)
encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

# Generate output with adjusted parameters for more deterministic output.
output_ids = model.generate(
    input_ids=encoded_input["input_ids"],
    attention_mask=encoded_input["attention_mask"],
    max_length=150,          # Shorter maximum length to prevent off-topic output.
    num_return_sequences=1,
    temperature=1,         # Lower temperature for more focused output.
    do_sample=True,
    top_p=0.9,
    # Optionally, you can try beam search:
    # num_beams=5, early_stopping=True
)

# Decode the generated text and truncate at the stop marker.
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
if "###" in generated_text:
    generated_text = generated_text.split("###")[0]

print("Generated Output:")
print(generated_text.strip())
