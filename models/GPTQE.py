import torch
from torch.nn import functional as F
from .model import GPT

class GPTQE(GPT):
    
    def forward(self, idx):
        device = idx.device
        batch_size, seq_len = idx.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # forward the GPT model itself
        token_emb = self.transformer.wte(idx)  # token embeddings of shape (batch_size, seq_len, n_embd)
        position_emb = self.transformer.wpe(pos)  # position embedding of shape (seq_len, n_embd)
        x = self.transformer.drop(token_emb + position_emb)  # dropout/regularization
        for block in self.transformer.h:  # applies each transformer block
            x = block(x)
        x = self.transformer.ln_f(x)  # final layer normalization layer
        logits = self.lm_head(x)  # convert layer to logits
        return logits

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]  # crops sequence to blocksize
            logits = self(idx_cond)  # forward the model to get logits forthe index in the sequence
            logits = logits[:, -1, :]  # pluck logits from final step
            logits[:, 0] = float('inf')  # sets logit of first token so probability 0sampled_ids)
            probs = F.softmax(-logits / temperature, dim=-1)  # apply softmax to get probabilities from logits 
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from probability distribution
            total_logits += torch.gather(logits, index=idx_next, dim=1)  # accumulates logits
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to sequence
        return idx, total_logits