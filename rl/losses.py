import torch
from torch.distributions import Categorical

def compute_ppo_loss(
    model, 
    tokens, 
    advantages, 
    epsilon, 
    old_model, 
    entropy_coef=0.01
):
    """
    Computes the Proximal Policy Optimization (PPO) clipped surrogate loss.

    Args:
        model: The current policy network.
        tokens: Tensor of generated sequences shape (batch_size, seq_len).
        advantages: Computed advantage estimates for the generated sequences.
        epsilon: Clipping parameter to bound the policy update.
        old_model: The reference policy network from the previous iteration.
        entropy_coef: Coefficient for the entropy bonus to encourage exploration.

    Returns:
        torch.Tensor: A scalar tensor representing the PPO loss.
    """

    # Policy evaluation on current and previous states
    dist = Categorical(logits=model(tokens[:, :-1]))
    curr_log_probs = dist.log_prob(tokens[:, 1:])

    with torch.no_grad():
        old_dist = Categorical(logits=old_model(tokens[:, :-1]))
        old_log_probs = old_dist.log_prob(tokens[:, 1:])

    # Importance sampling ratio with aggressive clamping to prevent NaNs
    ratios = torch.exp(torch.clamp(curr_log_probs - old_log_probs, min=-10.0, max=10.0))

    # Clipped surrogate objective
    surr1, surr2 = ratios * advantages, torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

    # Return negative loss for gradient descent, subtracting entropy bonus
    return -torch.mean(torch.min(surr1, surr2)) - (entropy_coef * dist.entropy().mean())


def compute_sil_loss(
    model, 
    sil_tokens, 
    sil_rewards, 
    floor=4.0, 
    scaling="linear"
):
    """
    Computes the Self-Imitation Learning (SIL) loss to reinforce high-reward past trajectories.

    Args:
        model: The current policy network.
        sil_tokens: Tensor of sequences sampled from the elite replay buffer.
        sil_rewards: Corresponding rewards for the sampled sequences.
        floor: The baseline reward threshold; only rewards above this floor yield positive advantage.
        scaling: Strategy for scaling advantages ("linear", "exponential", "z-score").

    Returns:
        torch.Tensor: A scalar tensor representing the SIL loss.
    """
    
    # Policy evaluation on current and previous states
    dist = Categorical(logits=model(sil_tokens[:, :-1]))
    sil_log_probs = dist.log_prob(sil_tokens[:, 1:])

    # Calculate raw advantages relative to the dynamic floor
    raw_advs = torch.clamp(sil_rewards - floor, min=0.1) 
    
    # Apply the selected scaling strategy to the advantages
    if scaling == "linear":
        advs = raw_advs * 2.0
    elif scaling == "exponential":
        advs = torch.exp(raw_advs / 1.0)
    elif scaling == "z-score":
        # Requires at least 2 elements to compute standard deviation
        advs = (raw_advs - raw_advs.mean()) / (raw_advs.std() + 1e-8) if len(raw_advs) > 1 else raw_advs
    else:
        raise ValueError(f"Unknown SIL scaling method: {scaling}")
        
    # Broadcast advantages to match sequence token shape and compute weighted log probabilities
    return -(sil_log_probs * advs.unsqueeze(-1).expand_as(sil_log_probs)).mean()

def compute_grpo_loss(
    model, 
    tokens, 
    advantages, 
    epsilon, 
    ref_model, 
    beta=0.01
):
    """
    Computes the Group Relative Policy Optimization (GRPO) loss with an exact KL penalty.

    Args:
        model: The current policy network.
        tokens: Tensor of generated sequences.
        advantages: Group-normalized advantage estimates.
        epsilon: Clipping parameter for the surrogate objective.
        ref_model: A static or slowly-updating reference policy network.
        beta: Coefficient scaling the KL divergence penalty.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The total GRPO loss and the mean KL divergence.
    """

    # Policy evaluation on current and previous states
    dist = Categorical(logits=model(tokens[:, :-1]))
    log_probs = dist.log_prob(tokens[:, 1:])

    with torch.no_grad():
        ref_dist = Categorical(logits=ref_model(tokens[:, :-1]))
        ref_log_probs = ref_dist.log_prob(tokens[:, 1:])

    # Calculate exact KL Divergence per token natively
    log_ratio = ref_log_probs - log_probs
    kl_div = torch.exp(log_ratio) - log_ratio - 1.0

    # PPO-style clipped surrogate objective using the reference model
    ratio = torch.exp(log_probs - ref_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # Total loss is the negative clipped surrogate plus the KL penalty
    loss = -torch.mean(torch.min(surr1, surr2) - beta * kl_div)
    
    # Return both the loss and the mean KL Divergence
    return loss, kl_div.mean()