import numpy as np
import torch
from typing import List
from collections import deque





def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[
                rand_ids
            ], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]
    


def compute_gae(
    next_value: list, 
    rewards: list, 
    masks: list, 
    values: list, 
    gamma: float, 
    tau: float
) -> List:
    """Compute gae.
    GAE help to reduce variance while maintaining a proper level of bias. By adjusting parameters λ∈[0,1]
    and γ∈[0,1]
    """
    values = values + [next_value]
    gae = 0
    returns: deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step]
            + gamma * values[step + 1] * masks[step]
            - values[step]
        )
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)            