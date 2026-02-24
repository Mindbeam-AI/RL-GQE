import torch
from torch.nn import functional as F
from model import GPT


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

    def calculate_loss(self, tokens, energies, b1):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # calculate the logits for the next possible tokens in the sequence
        logits = self(current_tokens)
        # get the logit for the actual next token in the sequence
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        # calculate the cumulative logits for each subsequence
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        # match cumulative logits to subsequence energies

        weights = 1 / (1 + torch.exp(b1 * energies))  # freeze gradient for true energy
        # weights = weights / weights.sum() * len(weights)

        # loss = torch.mean(weights * torch.exp(b2 * torch.abs(cumsum_logits - energies)))
        loss = torch.mean(weights * torch.square(cumsum_logits - energies))

        return loss

    def calculate_loss_GRPO(self, tokens, energies, epsilon, old_model=None):

        # --- Step 1: current log-probs ---
        logits = self(tokens[:, :-1])  # predict next-token logits
        log_probs = F.log_softmax(logits, dim=-1)
        curr_log_probs = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # --- Step 2: old log-probs (frozen policy) ---
        if old_model is not None:
            with torch.no_grad():
                old_logits = old_model(tokens[:, :-1])
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                old_log_probs = old_log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        else:
            old_log_probs = curr_log_probs.detach()

        # --- Step 3: rewards & normalized advantages (per token) with KL ---
        # Calculate KL divergence between current policy and old policy
        kl_div = torch.exp(old_log_probs) * (old_log_probs - curr_log_probs)
        
        # Introduce a KL coefficient (beta) to control the penalty strength (e.g., 0.05)
        beta_kl = 0.05
        
        # Subtract the KL penalty from the raw rewards BEFORE calculating advantages
        rewards = -energies
        penalized_rewards = rewards - (beta_kl * kl_div)
        
        # Calculate advantages using the penalized rewards
        mean_r = penalized_rewards.mean(dim=-1, keepdim=True)
        std_r = penalized_rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-4
        advantages = (penalized_rewards - mean_r) / std_r 
        
        #rewards = -energies                # [B, T-1] (skip start token)
        #mean_r = rewards.mean(dim=-1, keepdim=True)#mean_r = rewards.mean()
        #std_r = rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-4 #std_r = rewards.std(unbiased=False) + 1e-8
        # advantages = (rewards - mean_r) / std_r          # Â_{m,k}

        # --- Step 4: importance ratios & clipping ---
        log_ratio = curr_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
        ratios = torch.exp(log_ratio)
        #ratios_clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

        # Calculate unclipped and clipped surrogates
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        

        # --- Step 5: GRPO loss (Eq. 9 generalised to per-step rewards) ---
        #loss = -torch.mean(ratios_clipped * advantages)
        loss = -torch.mean(torch.min(surr1,surr2))
        
        # print({
        #     "r_mean": rewards.mean().item(),
        #     "r_std": rewards.std().item(),
        #     "A_mean": advantages.mean().item(),
        #     "A_std": advantages.std().item(),
        #     "ratio_mean": ratios.mean().item(),
        #     "ratio_std": ratios.std().item(),
        #     "loss": loss.item(),
        # })
        return loss, advantages

    def calculate_loss_DPO(self, tokens, energies, beta, ref_model=None):
        M, T = tokens.shape

        energies = energies.detach()
        energies = (energies - energies.mean()) / (energies.std() + 1e-6)

        logits = self(tokens[:, :-1])
        # ref_logits = ref_model(tokens[:, :-1])

        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        logp = token_logp.sum(dim=-1)

        logp_ref = -energies
        log_ratio = logp - logp_ref

        best_idx = energies.argmin()
        best_log_ratio = logp[best_idx] - logp_ref[best_idx]

        mask = torch.arange(M).to("cuda") != best_idx
        diff = best_log_ratio - log_ratio[mask]

        loss = F.softplus(-beta * diff)
        return loss.mean()

    def calculate_loss_CPO(self, tokens, energies, beta, ref_model=None):
        M, T = tokens.shape

        energies = energies.detach()
        energies = (energies - energies.mean()) / (energies.std() + 1e-6)

        logits = self(tokens[:, :-1])

        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        logp = token_logp.sum(dim=-1)          # (M,)

        # Boltzmann reference
        if ref_model is None:
            logp_ref = -energies                   # (M,)
        else:
            ref_logits = ref_model(tokens[:, :-1])
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_logp = ref_log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            logp_ref = ref_token_logp.sum(dim=-1)

        log_ratio = logp - logp_ref

        best_idx = energies.argmin()
        best_log_ratio = logp[best_idx] - logp_ref[best_idx]

        mask = torch.arange(M).to("cuda") != best_idx
        diff = best_log_ratio - log_ratio[mask]

        dpo_term = F.softplus(-beta * diff)
        loss = dpo_term - logp[best_idx]
        return loss.mean()

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