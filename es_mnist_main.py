from os import path
from typing import Any
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from cezo_fl.util import model_helpers
from cezo_fl.fl_helpers import get_server_name
from cezo_fl.util.metrics import Metric, accuracy
from cezo_fl.gradient_estimators.evolution_strategies_estimator import EvolutionStrategiesEstimator

from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    OptimizerSetting,
    ModelSetting,
    NormalTrainingLoopSetting,
    FrozenSetting,
)
from pydantic import Field, AliasChoices
from experiment_helper.device import use_device
from experiment_helper.data import (
    get_dataloaders,
    ImageClassificationTask,
)
from experiment_helper import prepare_settings


def train_model_es(epoch: int, model, train_loader, es_estimator, device, criterion, args, alpha: float) -> tuple[float, float]:
    """
    Train model using Evolution Strategies.
    
    ES Update Rule: θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
    where:
        - P is population size (num_pert)
        - α is learning rate (alpha)
        - r_norm_p is normalized reward for perturbation p
        - ε_p is the perturbation vector for p
    """
    model.train()
    train_loss = Metric("train loss")
    train_accuracy = Metric("train accuracy")
    iter_per_epoch = len(train_loader)
    
    with tqdm(total=iter_per_epoch, desc="Training:") as t, torch.no_grad():
        for iteration, (images, labels) in enumerate(train_loader):
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            
            # Generate random seed for this iteration
            seed = iteration**2 + iteration
            
            # Compute rewards for each perturbation in population
            # ES estimator's compute_grad returns rewards (negative loss)
            # loss_fn should take inputs and labels, and use the current (perturbed) model state
            def loss_fn(batch_inputs: torch.Tensor, batch_labels: torch.Tensor) -> torch.Tensor:
                outputs = model(batch_inputs)
                return criterion(outputs, batch_labels)
            
            # Store perturbations for later use in update (must match exactly)
            perturbations: list[dict[str, torch.Tensor]] = []
            rewards = []
            
            # Evaluate each perturbation
            for i in range(es_estimator.num_pert):
                rng = es_estimator.get_rng(seed, i)
                pert_dict: dict[str, torch.Tensor] = {}
                
                # Generate and store perturbation for each parameter
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    noise = torch.randn(
                        param.shape,
                        device=param.device,
                        dtype=param.dtype,
                        generator=rng
                    )
                    pert_dict[name] = noise
                    # Perturb model: θ + σ·ε
                    param.data.add_(noise, alpha=es_estimator.sigma)
                
                # Evaluate perturbed model (reward = negative loss)
                pert_loss = loss_fn(images, labels)
                reward = -pert_loss.item()  # Negative loss as reward (higher is better)
                rewards.append(reward)
                
                # Restore model: θ - σ·ε
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    param.data.add_(pert_dict[name], alpha=-es_estimator.sigma)
                
                perturbations.append(pert_dict)
            
            # Normalize rewards
            rewards_tensor = torch.tensor(rewards, device=device)
            rewards_mean = rewards_tensor.mean()
            rewards_std = rewards_tensor.std() + 1e-8
            rewards_normalized = (rewards_tensor - rewards_mean) / rewards_std
            
            # Update model using ES rule: θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
            # Use the SAME perturbations that were used for evaluation
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                update = torch.zeros_like(param)
                
                for i in range(es_estimator.num_pert):
                    # Use the exact same perturbation that was used for evaluation
                    noise = perturbations[i][name]
                    # Weight by normalized reward
                    update.add_(noise, alpha=float(rewards_normalized[i]))
                
                # Apply update: θ ← θ + (α/P) · update
                param.data.add_(update, alpha=alpha / es_estimator.num_pert)
            
            # Evaluate current model
            with torch.inference_mode():
                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = accuracy(outputs, labels)
            
            train_loss.update(loss.item())
            train_accuracy.update(acc.item())
            t.set_postfix({"Loss": train_loss.avg, "Accuracy": train_accuracy.avg})
            t.update(1)
    
    return train_loss.avg, train_accuracy.avg


def eval_model(epoch: int, model, test_loader, device, criterion) -> tuple[float, float]:
    """Evaluate model"""
    model.eval()
    eval_loss = Metric("Eval loss")
    eval_accuracy = Metric("Eval accuracy")
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader):
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            pred = model(images)
            eval_loss.update(criterion(pred, labels).item())
            eval_accuracy.update(accuracy(pred, labels).item())
    print(
        f"Evaluation(round {epoch}): Eval Loss:{eval_loss.avg:.4f}, "
        f"Accuracy:{eval_accuracy.avg * 100:.2f}%"
    )
    return eval_loss.avg, eval_accuracy.avg


class ESSetting(FrozenSetting):
    """ES-specific hyperparameters"""
    sigma: float = Field(
        default=0.01,
        description="Perturbation scale (sigma) for ES"
    )
    num_pert: int = Field(
        default=30,
        validation_alias=AliasChoices("num-pert"),
        description="Population size (number of perturbations) for ES"
    )
    alpha: float = Field(
        default=0.002,
        description="Learning rate (alpha) for ES update rule"
    )


class Setting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    OptimizerSetting,
    ModelSetting,
    NormalTrainingLoopSetting,
    ESSetting,
):
    """ES training settings"""
    pass


if __name__ == "__main__":
    args = Setting()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device_map = use_device(args.device_setting, 1)
    train_loaders, test_loader = get_dataloaders(
        args.data_setting, 1, args.seed, args.get_hf_model_name()
    )
    train_loader = train_loaders[0]
    device = device_map[get_server_name()]
    
    criterion = torch.nn.CrossEntropyLoss()
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    model = prepare_settings.get_model(args.dataset, args.model_setting, args.seed).to(device)
    
    # ES estimator
    es_estimator = EvolutionStrategiesEstimator(
        model.parameters(),
        sigma=args.sigma,
        num_pert=args.num_pert,
        device=device,
        torch_dtype=args.model_setting.get_torch_dtype(),
    )
    
    if args.log_to_tensorboard:
        tensorboard_sub_folder = model.model_name + "-" + model_helpers.get_current_datetime_str()
        writer = SummaryWriter(
            path.join(
                "tensorboards",
                args.dataset.value,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )
    
    print(f"🔹 Evolution Strategies Training")
    print(f"Population Size: {args.num_pert}, Sigma: {args.sigma}, Alpha: {args.alpha}")
    print(f"Dataset: {args.dataset.value}, Epochs: {args.epoch}")
    print("")
    
    for epoch in range(args.epoch):
        train_loss, train_accuracy = train_model_es(
            epoch, model, train_loader, es_estimator, device, criterion, args, args.alpha
        )
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        eval_loss, eval_accuracy = eval_model(epoch, model, test_loader, device, criterion)
        if args.log_to_tensorboard:
            writer.add_scalar("Loss/test", eval_loss, epoch)
            writer.add_scalar("Accuracy/test", eval_accuracy, epoch)
    
    if args.log_to_tensorboard:
        writer.close()
    
    print("")
    print("✅ ES Training completed!")
