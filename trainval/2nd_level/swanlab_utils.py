import os
from datetime import datetime


class SwanLabLogger:
    def __init__(self, script_name, config=None):
        self.enabled = os.environ.get("ENABLE_SWANLAB", "0") == "1"
        self._run = None
        if not self.enabled:
            return

        try:
            import swanlab

            project = os.environ.get("SWANLAB_PROJECT", "rsna-pe-repro")
            experiment_name = os.environ.get(
                "SWANLAB_EXPERIMENT_NAME",
                f"{script_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            self._run = swanlab
            self._run.init(
                project=project,
                experiment_name=experiment_name,
                config=config or {},
            )
        except Exception as exc:
            self.enabled = False
            print(f"[WARN] SwanLab disabled: {exc}", flush=True)

    def log_epoch(self, epoch, train_losses, valid_losses, kaggle_loss, weighted_metrics, overall_metrics, label_metrics, lr=None):
        if not self.enabled or self._run is None:
            return
        try:
            payload = {"epoch": epoch, "kaggle_loss": float(kaggle_loss)}
            if lr is not None:
                payload["learning_rate"] = float(lr)

            for key, value in train_losses.items():
                payload[f"train/{key}"] = float(value)
            for key, value in valid_losses.items():
                payload[f"valid/{key}"] = float(value)
            for key, value in weighted_metrics.items():
                payload[f"weighted/{key}"] = float(value)
            if overall_metrics is not None:
                for key, value in overall_metrics.items():
                    if isinstance(value, (int, float)):
                        payload[f"overall_pe/{key}"] = float(value)
            for label_name, metrics in label_metrics.items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        payload[f"labels/{label_name}/{key}"] = float(value)

            self._run.log(payload, step=epoch)
        except Exception as exc:
            self.enabled = False
            print(f"[WARN] SwanLab logging disabled after runtime error: {exc}", flush=True)

    def finish(self):
        if not self.enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            print(f"[WARN] SwanLab finish failed: {exc}", flush=True)
