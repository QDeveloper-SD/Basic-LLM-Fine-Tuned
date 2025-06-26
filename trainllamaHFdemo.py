from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import pandas as pd

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf" # change to your specific LLaMA variant
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Enable PEFT with LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# === Your dataset goes here ===
# Replace this with your actual dataset
# Example: dataset = load_dataset("your_dataset_name")
pf = 'FilePath'
pfd = pd.read_csv(pf)
# df1 = pd.DataFrame(pfd)
dataset = load_dataset(pfd)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation"),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# === Inference after training ===
model.eval()

input_text = "Which methods am I able to use to evaluate a stock market portfolio?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Automatically determine available space for generation
max_context_length = model.config.max_position_embeddings # usually 2048 or 4096
input_length = inputs["input_ids"].shape[1]
max_new_tokens = max_context_length - input_length

# Ensure some reasonable minimum generation
min_tokens = 20
if max_new_tokens < min_tokens:
    max_new_tokens = min_tokens

# Generate and decode output
output = model.generate(**inputs, max_new_tokens=max_new_tokens)
print(tokenizer.decode(output[0], skip_special_tokens=True))
