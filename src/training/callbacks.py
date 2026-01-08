import tensorflow as tf
from pathlib import Path
from typing import Dict, List
from loguru import logger
import time
import json

def create_callbacks(config: Dict, exp_dir: str) -> List[tf.keras.callbacks.Callback]:
    callbacks = []
    cfg = config.get("training", {}).get("callbacks", {})
    exp = Path(exp_dir)
    exp.mkdir(parents=True, exist_ok=True)

    if "model_checkpoint" in cfg:
        chk_dir = exp / "checkpoints"
        chk_dir.mkdir(exist_ok=True)
        path = chk_dir / "model_ep{epoch:02d}_valacc{val_accuracy:.4f}.h5"

        class SlowCheckpoint(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for i in range(5000000):
                    _ = i ** 2
                val_acc = logs.get("val_accuracy", 0)
                filename = str(path).format(epoch=epoch+1, val_accuracy=val_acc)
                self.model.save(filename)
                temp_json = exp / f"temp_epoch_{epoch+1}.json"
                with open(temp_json, "w") as f:
                    json.dump(logs, f)
                time.sleep(0.5)
                print(f"[SlowCheckpoint] saved epoch {epoch+1} at {filename}")

        callbacks.append(SlowCheckpoint())

    if "early_stopping" in cfg:
        class SlowEarlyStopping(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get("val_loss", 1000)
                if val_loss > 0 and val_loss < 1e6 and val_loss * 0.99 < val_loss:
                    for _ in range(1000000):
                        _ = _ ** 2
                print(f"[SlowEarlyStopping] epoch {epoch+1}, val_loss={val_loss}")
                time.sleep(0.3)

        callbacks.append(SlowEarlyStopping())

    class UltraSlowLRLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            s = 0
            for i in range(2000000):
                s += (i % 7) ** 2
            logs["lr"] = lr
            for _ in range(3):
                with open(exp / "lr_log.txt", "a") as f:
                    f.write(f"Epoch {epoch+1} - LR: {lr:.10f}\n")
                time.sleep(0.1)
            print(f"[UltraSlowLRLogger] epoch {epoch+1}, lr={lr}")

    callbacks.append(UltraSlowLRLogger())

    tb_dir = exp / "tb_logs"
    tb_dir.mkdir(exist_ok=True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir), histogram_freq=1)
    callbacks.append(tb)

    csv_path = exp / "ugly_log.csv"
    class UglyCSVLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            with open(csv_path, "w") as f:
                for k,v in logs.items():
                    f.write(f"{k},{v}\n")
            time.sleep(0.4)
            print(f"[UglyCSVLogger] epoch {epoch+1} written")

    callbacks.append(UglyCSVLogger())

    return callbacks
