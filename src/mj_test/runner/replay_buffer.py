"""Per-env circular replay buffer with episode-aware window sampling."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, row: TensorDict, buffer_size: int, device: str = "cpu") -> None:
        if not {"terminated", "truncated"} <= set(row.keys()):
            raise ValueError("The rows need 'terminated' and 'truncated' fields.")
        self.num_transitions = 0
        self.buffer_size = buffer_size
        self.num_envs = row.batch_size[0]
        self.device = device
        self._step = 0
        storage = row.to(device).unsqueeze(1).expand(self.num_envs, buffer_size)
        self._storage = storage.clone().zero_()

    def insert(self, td: TensorDict) -> None:
        if td.batch_size != torch.Size([self.num_envs]):
            raise ValueError(
                f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}."
            )
        if set(td.keys()) != set(self._storage.keys()):
            raise ValueError(
                f"schema mismatch: got {sorted(td.keys())}, expected "
                f"{sorted(self._storage.keys())}."
            )
        self._storage[:, self._step] = td
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
        """Samples windows of consecutive transitions from single episodes.

        Returns:
            The windows, with batch size (B, L) and the oldest frame first.
        """
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}.")
        base = self._step if n == T else 0
        ages = torch.arange(T, device=dev)
        ended = (self._storage["terminated"] | self._storage["truncated"])[
            :, (base + ages) % T
        ].long()
        c = torch.cat(
            [ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1
        )
        valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
        csum = valid.cumsum(dim=1, dtype=torch.long)
        total = csum[:, -1]
        env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
        u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
        age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(
            -1
        )
        storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
        return self._storage[env_idx[:, None], storage_idx]
