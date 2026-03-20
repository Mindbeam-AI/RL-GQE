import torch

class EliteReplayBuffer:
    def __init__(self, max_size=100, absolute_floor=4.0):
        self.max_size = max_size
        self.absolute_floor = absolute_floor
        self.tokens = None
        self.rewards = None

    def add(self, new_tokens, new_rewards, advantages):
        curr_floor = max(self.absolute_floor, self.rewards[-1].item()) if self.is_full() else self.absolute_floor
        mask = (advantages > 0) & (new_rewards > curr_floor)
        valid_tokens, valid_rewards = new_tokens[mask], new_rewards[mask]

        if len(valid_tokens) == 0: return

        if self.tokens is None:
            self.tokens, self.rewards = valid_tokens, valid_rewards
        else:
            self.tokens = torch.cat([self.tokens, valid_tokens], dim=0)
            self.rewards = torch.cat([self.rewards, valid_rewards], dim=0)

        sorted_indices = torch.argsort(self.rewards, descending=True)
        self.tokens, self.rewards = self.tokens[sorted_indices][:self.max_size], self.rewards[sorted_indices][:self.max_size]

    def is_full(self):
        return self.tokens is not None and len(self.tokens) >= self.max_size

    def sample(self, batch_size):
        if not self.is_full(): return None, None
        idx = torch.randperm(len(self.tokens), device=self.tokens.device)[:min(len(self.tokens), batch_size)]
        return self.tokens[idx], self.rewards[idx]

    def stats(self):
        if self.tokens is None: return 0, 0.0, 0.0
        return len(self.tokens), self.rewards.mean().item(), self.rewards[0].item()