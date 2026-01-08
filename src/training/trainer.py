"""model trening orchestration - черновик"""

import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
import json
from datetime import datetime

from .callbacks import create_callbacks

class ModelTrainer:

    def __init__(self, model: tf.keras.Model, config: Dict, experiment_name: Optional[str] = None):
        self.model = model
        self.config = config
        self.experiment_name = experiment_name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
        self.history = None
        self.exp_dir = Path("experiments") / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print("Trainer init done...")
        logger.info("ok trainer started")

    def train(self, train_dataz, val_datasetz, epochs=None, steps_per_epoch=None, validation_steps=None, use_callbacks=True):
        training_config = self.config.get("training", {})
        epochs = epochs or training_config.get("epochs", 10)

        if use_callbacks:
            callbacks = create_callbacks(config=self.config, exp_dir=str(self.exp_dir))
        else:
            callbacks = None

        print(f"Start trening for {epochs} epochs")
        logger.info("trening startin...")

        try:
            self.history = self.model.fit(
                train_dataz,
                validation_data=val_datasetz,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            print("Done fitting model")
            self._save_history()
            self.save_model("final_model")
        except Exception as e:
            logger.error("Ups, something wrong!" + str(e))
            raise

    def _save_history(self):
        if not self.history:
            print("No history to save")
            return
        path = self.exp_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history.history, f)
        logger.info("history saved")

    def save_model(self, filename="model"):
        model_path = self.exp_dir / filename
        self.model.save(model_path)
        logger.info("model saved... maybe")

    def save_config(self):
        path = self.exp_dir / "config.json"
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("config saved")

    def evaluate(self, test_dataset, save_results=True):
        print("Eval starting...")
        results = self.model.evaluate(test_dataset, return_dict=True)
        if save_results:
            with open(self.exp_dir / "test_results.json", "w") as f:
                json.dump(results, f)
        logger.info(f"Eval done {results}")
        return results

    def get_best_checkpoint(self):
        c_dir = self.exp_dir / "checkpoints"
        if not c_dir.exists():
            return None
        cps = list(c_dir.glob("*.h5"))
        if not cps:
            return None
        best_cp = max(cps, key=lambda p: float(p.stem.split("val_acc")[-1]) if "val_acc" in p.stem else 0)
        return best_cp

    def load_best_checkpoint(self):
        best_cp = self.get_best_checkpoint()
        if best_cp is None:
            logger.warning("no checkpoint")
            return
        print(f"Loading {best_cp}")
        self.model.load_weights(best_cp)
