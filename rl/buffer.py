import torch

class EliteReplayBuffer:
    """
    A priority replay buffer designed for Self-Imitation Learning (SIL).
    
    This buffer maintains a dynamically sizing queue of the highest-reward trajectories 
    (elite circuits) encountered during training. It filters incoming trajectories based 
    on positive advantages and a dynamic reward floor to ensure only optimal topologies are retained.
    """
    
    def __init__(self, max_size=100, absolute_floor=4.0):
        """
        Args:
            max_size: The maximum number of trajectories to store. 
            absolute_floor: The minimum baseline reward a trajectory must achieve to be considered.
        """
        self.max_size = max_size
        self.absolute_floor = absolute_floor

        # Initialized as None to save memory before the first valid trajectories are found
        self.tokens = None
        self.rewards = None

    def add(self, new_tokens, new_rewards, advantages):
        """
        Evaluates and integrates new generated sequences into the elite buffer.
        Only sequences that outperform the group average (advantages > 0) AND 
        beat the current dynamic floor are added.
        
        Args:
            new_tokens: Tensor of generated sequences.
            new_rewards: Tensor of rewards corresponding to the sequences.
            advantages: Tensor of advantages (e.g., GRPO/PPO advantage estimates).
        """

        # The dynamic floor ensures we only accept sequences better than the worst sequence currently in a full buffer
        curr_floor = max(self.absolute_floor, self.rewards[-1].item()) if self.is_full() else self.absolute_floor
        
        # Mask filters for strictly advantageous and high-reward trajectories
        mask = (advantages > 0) & (new_rewards > curr_floor)
        valid_tokens, valid_rewards = new_tokens[mask], new_rewards[mask]

        if len(valid_tokens) == 0: return

        # Initialize or concatenate tensors
        if self.tokens is None:
            self.tokens, self.rewards = valid_tokens, valid_rewards
        else:
            self.tokens = torch.cat([self.tokens, valid_tokens], dim=0)
            self.rewards = torch.cat([self.rewards, valid_rewards], dim=0)

        # Sort descending by reward and enforce the capacity limit
        sorted_indices = torch.argsort(self.rewards, descending=True)
        self.tokens, self.rewards = self.tokens[sorted_indices][:self.max_size], self.rewards[sorted_indices][:self.max_size]

    def is_full(self):
        """
        Checks if the buffer has reached its designated capacity.
        
        Returns:
            bool: True if the buffer holds `max_size` trajectories, False otherwise.
        """
        
        return self.tokens is not None and len(self.tokens) >= self.max_size

    def sample(self, batch_size):
        """
        Randomly samples a batch of elite trajectories for self-imitation updates.
        
        Args:
            batch_size: The number of trajectories to sample.
            
        Returns:
            Tuple containing the sampled tokens and rewards tensors. Returns (None, None) if not full.
        """
        
        if not self.is_full(): return None, None

        # randperm is a faster and safer than random choice for tensor indexing on GPU
        idx = torch.randperm(len(self.tokens), device=self.tokens.device)[:min(len(self.tokens), batch_size)]
        return self.tokens[idx], self.rewards[idx]

    def stats(self):
        """
        Retrieves the current state metrics of the buffer.
        
        Returns:
            Tuple[int, float, float]: Current size, mean reward, and maximum reward (the elite sequence).
        """
        
        if self.tokens is None: return 0, 0.0, 0.0
        return len(self.tokens), self.rewards.mean().item(), self.rewards[0].item()

    def shrink_capacity(self, new_max_size):
        """
        Dynamically reduces the capacity of the buffer. This is used to force the model 
        to fixate on an increasingly narrower band of elite sequences as training progresses.
        
        Args:
            new_max_size: The new maximum capacity constraint.
        """
        
        self.max_size = new_max_size
        
        # Because add() maintains descending order, slicing the top elements preserves the highest rewards
        if self.tokens is not None and len(self.tokens) > new_max_size:
            # Tensors are already sorted descending in add(), so we just keep the top slice
            self.tokens = self.tokens[:new_max_size]
            self.rewards = self.rewards[:new_max_size]