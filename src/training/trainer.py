from pathlib import Path

import torch
from tqdm.auto import tqdm


class Trainer:
    """Train a model and use the evaluator for validation."""

    def __init__(
        self,
        model,
        train_dataloader,
        optimizer,
        loss_function,
        evaluator,
        device="cpu",
        epochs=100,
        early_stopping_patience=None,
        checkpoint_path=None,
        mixed_precision=False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.evaluator = evaluator
        self.device = torch.device(device)
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.mixed_precision = resolve_mixed_precision(mixed_precision, self.device)
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.mixed_precision,
        )

    def train(self):
        """Train for multiple epochs and return the collected history."""
        self.model.to(self.device)
        history = []
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        print(
            "Mixed precision: "
            f"{'enabled' if self.mixed_precision else 'disabled'} "
            f"(device: {self.device.type})"
        )

        epoch_progress = tqdm(
            range(1, self.epochs + 1),
            desc="Epochs",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            train_loss = self._train_epoch(epoch)

            # Validation is delegated to Evaluator to keep metric logic outside Trainer.
            val_metrics = self.evaluator.evaluate_paired(
                progress_description=f"Val   {epoch}/{self.epochs}"
            )
            val_loss = val_metrics.get("loss")

            epoch_results = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{name.lower()}": value for name, value in val_metrics.items()},
            }

            self._print_epoch_summary(epoch_results)
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}" if val_loss is not None else "n/a",
            )

            history.append(epoch_results)

            # Early stopping needs validation loss, so it is skipped if no loss was given.
            if val_loss is None:
                continue

            # The checkpoint stores the model with the best validation loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                epochs_without_improvement += 1

            if self._should_stop_early(epochs_without_improvement):
                break

        return history

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        sample_count = 0

        progress = tqdm(
            self.train_dataloader,
            desc=f"Train {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in progress:
            # Training datasets return paired low-light and normal-light images.
            low_images = batch["low"].to(self.device)
            high_images = batch["high"].to(self.device)
            batch_size = low_images.shape[0]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.mixed_precision,
            ):
                predictions = self.model(low_images)
                loss = self.loss_function(predictions, high_images)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Weighting by batch size handles a smaller final batch correctly.
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total_loss / sample_count:.4f}",
            )

        if sample_count == 0:
            raise ValueError("Cannot train with an empty dataloader")

        return total_loss / sample_count

    def _should_stop_early(self, epochs_without_improvement):
        return (
            self.early_stopping_patience is not None
            and epochs_without_improvement >= self.early_stopping_patience
        )

    def _save_checkpoint(self, epoch, val_metrics):
        if self.checkpoint_path is None:
            return

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
            },
            self.checkpoint_path,
        )

    def _print_epoch_summary(self, results):
        parts = [
            f"Epoch {results['epoch']}/{self.epochs}",
            f"train_loss={results['train_loss']:.4f}",
        ]

        if "val_loss" in results:
            parts.append(f"val_loss={results['val_loss']:.4f}")

        if "val_psnr" in results:
            parts.append(f"val_psnr={results['val_psnr']:.2f}")

        if "val_ssim" in results:
            parts.append(f"val_ssim={results['val_ssim']:.4f}")

        print(" | ".join(parts))


def resolve_mixed_precision(value, device):
    """Enable mixed precision automatically only where it is reliable."""
    device = torch.device(device)
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value == "auto":
            return device.type == "cuda"
        if normalized_value in {"true", "yes", "1"}:
            return device.type == "cuda"
        if normalized_value in {"false", "no", "0"}:
            return False
        raise ValueError(f"Unsupported mixed_precision value: {value}")

    return bool(value) and device.type == "cuda"
