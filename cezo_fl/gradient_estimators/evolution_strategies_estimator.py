from typing import Callable, Iterator, Sequence
import torch
from torch.nn import Parameter
from cezo_fl.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator


class EvolutionStrategiesEstimator(AbstractGradientEstimator):
    """
    Evolution Strategies estimator for direct reward optimization.
    Unlike ZO gradient estimation, ES directly optimizes rewards without
    computing gradients. Uses reward-weighted perturbations.

    ES Update Rule:
        θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
    where:
        - P is population size (num_pert)
        - α is learning rate (sigma)
        - r_norm_p is normalized reward for perturbation p
        - ε_p is the perturbation vector for p
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        sigma: float = 0.001,
        num_pert: int = 30,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.parameters_list: list[Parameter] = [p for p in parameters if p.requires_grad]
        self.total_dimensions = sum([p.numel() for p in self.parameters_list])
        print(f"ES trainable model size: {self.total_dimensions}")

        self.sigma = sigma  # Perturbation scale (like mu in ZO)
        self.num_pert = num_pert  # Population size
        self.device = device
        self.torch_dtype = torch_dtype

    def get_rng(self, seed: int, perturb_index: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            seed * (perturb_index + 17) + perturb_index
        )

    def generate_perturbation_norm(self, rng: torch.Generator | None = None) -> torch.Tensor:
        """Generate random perturbation vector (standard normal)"""
        p = torch.randn(
            self.total_dimensions, device=self.device, dtype=self.torch_dtype, generator=rng
        )
        return p

    def perturb_model(self, perturb: torch.Tensor | None = None, alpha: float | int = 1) -> None:
        """Add perturbation to model parameters"""
        start = 0
        for p in self.parameters_list:
            if perturb is not None:
                _perturb = perturb[start : (start + p.numel())]
                p.add_(_perturb.view(p.shape), alpha=alpha)
            start += p.numel()

    def compute_grad(
        self,
        batch_inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int,
    ) -> torch.Tensor:
        """
        Compute rewards for ES (returns rewards as 'gradients' for compatibility).
        For ES, we return rewards instead of gradient scalars.
        Higher reward = better performance (we use negative loss as reward).
        """
        rewards = []

        for i in range(self.num_pert):
            rng = self.get_rng(seed, i)
            pb_norm = self.generate_perturbation_norm(rng)

            # Perturb model: θ + σ·ε
            self.perturb_model(pb_norm, alpha=self.sigma)

            # Evaluate perturbed model (reward = negative loss)
            pert_loss = loss_fn(batch_inputs, labels)
            reward = -pert_loss.item()  # Negative loss as reward (higher is better)
            rewards.append(reward)

            # Restore model: θ - σ·ε
            self.perturb_model(pb_norm, alpha=-self.sigma)

            del pb_norm

        # Return rewards as tensor (compatible with gradient scalar interface)
        return torch.tensor(rewards, device=self.device, dtype=self.torch_dtype)

    def update_gradient_estimator_given_seed_and_grad(
        self,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        """No updates needed for ES estimator state"""
        pass

    def update_model_given_seed_and_grad(
        self,
        optimizer: torch.optim.Optimizer,
        iteration_seeds: Sequence[int],
        iteration_grad_scalar: Sequence[torch.Tensor],
    ) -> None:
        """
        Update model using ES rule: θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)

        Args:
            optimizer: Not used in ES (we update directly), but kept for interface compatibility
            iteration_seeds: List of seeds for perturbations
            iteration_grad_scalar: List of reward tensors (one per seed)
        """
        assert len(iteration_seeds) == len(iteration_grad_scalar)

        # Get learning rate from optimizer (alpha in ES update rule)
        alpha = optimizer.defaults["lr"]  # ES learning rate

        # Aggregate rewards across all iterations
        all_rewards: list[float] = []
        all_seeds: list[tuple[int, int]] = []

        # Aggregate rewards and track seeds properly
        for iteration_seed, reward_tensor in zip(iteration_seeds, iteration_grad_scalar):
            # reward_tensor is shape [num_pert] containing rewards for each perturbation
            rewards_list = reward_tensor.cpu().tolist()
            all_rewards.extend(rewards_list)
            # Store (iteration_seed, perturbation_index) pairs for correct seed generation
            for i in range(len(rewards_list)):
                all_seeds.append((iteration_seed, i))

        if len(all_rewards) == 0:
            return

        # Normalize rewards
        rewards_tensor = torch.tensor(all_rewards, device=self.device, dtype=self.torch_dtype)
        rewards_mean = rewards_tensor.mean()
        rewards_std = rewards_tensor.std() + 1e-8
        rewards_normalized = (rewards_tensor - rewards_mean) / rewards_std

        # ES Update: θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
        # Generate full-dimensional perturbations (same as client) and split across parameters
        start = 0
        for param in self.parameters_list:
            update = torch.zeros_like(param)

            for seed_idx, (iteration_seed, pert_idx) in enumerate(all_seeds):
                # Generate full perturbation vector using SAME seed generation as client
                rng = self.get_rng(iteration_seed, pert_idx)  # Must match client: get_rng(seed, i)
                full_noise = self.generate_perturbation_norm(rng)
                # Extract the portion for this parameter
                param_noise = full_noise[start : (start + param.numel())].view(param.shape)
                # Weight by normalized reward
                update.add_(param_noise, alpha=float(rewards_normalized[seed_idx]))

            # Apply update: θ ← θ + (α/P) · update
            # P is the population size (num_pert), not total_perturbations
            # The normalization already accounts for aggregating across multiple local steps
            # This matches the non-FL ES update rule: θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
            param.data.add_(update, alpha=alpha / self.num_pert)
            start += param.numel()
