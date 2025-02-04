import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Check for GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the model name (using the MBZUAI/LaMini-GPT-1.5B checkpoint)
model_name = "MBZUAI/LaMini-GPT-1.5B"

# Load the tokenizer and model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# ------------------------------
# Prepare Your Tennis Team Dataset
# ------------------------------
# In this example, we create a small dataset.
# In your actual use case, replace this with your full dataset.
data = [
    {
        "player_id": 1,
        "ad_side_performance": 0.85,
        "deuce_side_performance": 0.75,
        "rating": 3.0,
        "match_details": "Won 6-4, 6-3",
        "compatible_partners": "Player 2, Player 3"  # Expected output or label
    },
    {
        "player_id": 2,
        "ad_side_performance": 0.78,
        "deuce_side_performance": 0.82,
        "rating": 3.2,
        "match_details": "Lost 5-7, 6-4, 6-7",
        "compatible_partners": "Player 1, Player 4"
    },
    {
        "player_id": 3,
        "ad_side_performance": 0.90,
        "deuce_side_performance": 0.70,
        "rating": 3.1,
        "match_details": "Won 6-3, 6-2",
        "compatible_partners": "Player 1, Player 4"
    },
    {
        "player_id": 4,
        "ad_side_performance": 0.80,
        "deuce_side_performance": 0.85,
        "rating": 3.3,
        "match_details": "Won 7-5, 6-4",
        "compatible_partners": "Player 2, Player 3"
    },
]

print("Raw Data:", data)

# Convert the list of dictionaries to a Hugging Face Dataset.
dataset = Dataset.from_dict({
    "player_id": [d["player_id"] for d in data],
    "ad_side_performance": [d["ad_side_performance"] for d in data],
    "deuce_side_performance": [d["deuce_side_performance"] for d in data],
    "rating": [d["rating"] for d in data],
    "match_details": [d["match_details"] for d in data],
    "compatible_partners": [d["compatible_partners"] for d in data],
})

print("Dataset:", dataset)

# ------------------------------
# Tokenization
# ------------------------------
def tokenize_function(examples):
    # Create a formatted prompt string for each record.
    texts = [
        (
            f"Player ID: {pid}\n"
            f"Ad Side Performance: {ad}\n"
            f"Deuce Side Performance: {deuce}\n"
            f"Rating: {rating}\n"
            f"Match Details: {details}\n"
            f"Compatible Partners: {partners}\n"
            "###\n"  # Delimiter indicating the end of the record
        )
        for pid, ad, deuce, rating, details, partners in zip(
            examples["player_id"],
            examples["ad_side_performance"],
            examples["deuce_side_performance"],
            examples["rating"],
            examples["match_details"],
            examples["compatible_partners"],
        )
    ]
    # Tokenize the texts
    tokenized_output = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    # Set the labels to be the same as the input_ids so that the model can compute the loss.
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

# Apply tokenization to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("Tokenized Dataset:", tokenized_dataset)

# ------------------------------
# Split the Dataset into Train and Evaluation Sets
# ------------------------------
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("Training Dataset:", train_dataset)
print("Evaluation Dataset:", eval_dataset)

# ------------------------------
# Define Training Arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,                    # The learning rate to be tuned
    per_device_train_batch_size=2,         # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=1,                    # Increase for real training
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",                      # Change to "tensorboard" if desired
)

# ------------------------------
# Define the Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# ------------------------------
# Fine-tune the Model
# ------------------------------
print("Starting training...")
trainer.train()

# Save the fine-tuned model and tokenizer
model_save_path = "./fine-tuned-la-mini-gpt-1.5b"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")

# ------------------------------
# Evaluate the Model (Optional)
# ------------------------------
print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
