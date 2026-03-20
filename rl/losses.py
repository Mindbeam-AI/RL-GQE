import torch
from torch.distributions import Categorical

def compute_ppo_loss(model, tokens, advantages, epsilon, old_model, entropy_coef=0.01):
    dist = Categorical(logits=model(tokens[:, :-1]))
    curr_log_probs = dist.log_prob(tokens[:, 1:])

    with torch.no_grad():
        old_dist = Categorical(logits=old_model(tokens[:, :-1]))
        old_log_probs = old_dist.log_prob(tokens[:, 1:])

    ratios = torch.exp(torch.clamp(curr_log_probs - old_log_probs, min=-10.0, max=10.0))
    surr1, surr2 = ratios * advantages, torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
    
    return -torch.mean(torch.min(surr1, surr2)) - (entropy_coef * dist.entropy().mean())

def compute_sil_loss(model, sil_tokens, sil_rewards, floor=4.0):
    dist = Categorical(logits=model(sil_tokens[:, :-1]))
    sil_log_probs = dist.log_prob(sil_tokens[:, 1:])

    raw_advs = torch.clamp(sil_rewards - floor, min=0.1) 
    advs = (raw_advs - raw_advs.mean()) / (raw_advs.std() + 1e-8) if len(raw_advs) > 1 else raw_advs

    return -(sil_log_probs * advs.unsqueeze(-1).expand_as(sil_log_probs)).mean()