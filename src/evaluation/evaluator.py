from collections import defaultdict

import torch
from tqdm.auto import tqdm


class Evaluator:
    """Run paired or unpaired evaluation over a complete dataset."""

    def __init__(self, model, dataloader, metric_calculator, device="cpu", loss_function=None):
        self.model = model
        self.dataloader = dataloader
        self.metric_calculator = metric_calculator
        self.device = torch.device(device)
        self.loss_function = loss_function

    def evaluate_paired(self, progress_description=None):
        """Evaluate images with a normal-light ground truth using reference metrics."""
        return self._evaluate(
            input_key="low",
            target_key="high",
            progress_description=progress_description,
        )

    def evaluate_unpaired(self, progress_description=None):
        """Evaluate images without ground truth using no-reference metrics."""
        return self._evaluate(
            input_key="image",
            target_key=None,
            progress_description=progress_description,
        )

    def _evaluate(self, input_key, target_key, progress_description=None):
        self.model.to(self.device)

        # Evaluation mode disables training behavior such as dropout updates.
        self.model.eval()
        metric_totals = defaultdict(float)
        sample_count = 0

        # Gradients are not needed during validation or test evaluation.
        with torch.inference_mode():
            progress = tqdm(
                self.dataloader,
                desc=progress_description,
                leave=False,
                dynamic_ncols=True,
                disable=progress_description is None,
            )
            for batch in progress:
                inputs = batch[input_key].to(self.device)
                targets = batch[target_key].to(self.device) if target_key else None
                predictions = self.model(inputs)
                batch_size = inputs.shape[0]

                # MetricCalculator decides which metrics are available from the target.
                scores = self.metric_calculator.calculate(
                    predictions,
                    targets,
                    inputs,
                )
                for metric_name, value in scores.items():
                    # Weighted sums give correct averages when the last batch is smaller.
                    metric_totals[metric_name] += value * batch_size

                # Validation loss is optional, while test evaluation may only need metrics.
                if self.loss_function is not None and targets is not None:
                    loss = self.loss_function(predictions, targets)
                    metric_totals["loss"] += loss.item() * batch_size

                sample_count += batch_size
                if "loss" in metric_totals:
                    progress.set_postfix(
                        loss=f"{metric_totals['loss'] / sample_count:.4f}"
                    )

        if sample_count == 0:
            raise ValueError("Cannot evaluate an empty dataloader")

        return {
            metric_name: total / sample_count
            for metric_name, total in metric_totals.items()
        }
