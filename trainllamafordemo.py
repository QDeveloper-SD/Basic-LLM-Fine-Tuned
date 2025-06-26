import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# ADD the dynamic max_new_tokens code from the HF demo
# === Load and Prepare Data ===
df = pd.read_csv("your_dataset.csv")  # Must have a "text" column
text = " ".join(df["text"].dropna().astype(str).tolist())

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])

# Encode entire text
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# === Model Hyperparameters ===
block_size = 128
batch_size = 64
embedding_dim = 128
n_heads = 4
n_layers = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Data Loader ===
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# === Simple Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        return self.ln2(x + ff_out)

# === Full LLM Model ===
class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# === Training Loop ===
model = LLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(500):
    model.train()
    xb, yb = get_batch("train")
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# === Generate Text ===
context = torch.zeros((1, 1), dtype=torch.long).to(device)  # start with null token
generated = model.generate(context, max_new_tokens=200)
print("Generated Text:\n", decode(generated[0].tolist()))


# === Save the model ===
torch.save(model.state_dict(), "llm_model.pt")
print("Model saved to llm_model.pt")

# === Reload the model ===
# To load later or in another script:
def load_model():
    model = LLM().to(device)
    model.load_state_dict(torch.load("llm_model.pt", map_location=device))
    model.eval()
    return model

# === Prompt for generation ===
user_input = input("Enter a prompt: ").strip()
if not user_input:
    print("No prompt provided. Using empty context.")
    user_input = " "

# Encode prompt
start_ids = torch.tensor([encode(user_input)], dtype=torch.long).to(device)

# Load model and generate
model = load_model()
with torch.no_grad():
    output = model.generate(start_ids, max_new_tokens=200)
    generated_text = decode(output[0].tolist())

print("\n=== Generated Text ===")
print(generated_text)