import json
import os
import matplotlib.pyplot as plt

METRICS_PATH = "./deepspeed_ws/checkpoints/metrics_epoch.jsonl"
OUT_DIR = "./deepspeed_ws/analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)

epochs = []
train_loss = []
val_loss = []
epoch_time = []
samples_per_sec_global = []
peak_mem_gb = []

with open(METRICS_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        epochs.append(rec["epoch"])
        train_loss.append(rec["train_loss"])
        val_loss.append(rec["val_loss"])
        epoch_time.append(rec["epoch_time"])
        samples_per_sec_global.append(rec["samples_per_sec_global"])
        peak_mem_gb.append(rec.get("peak_mem_gb_rank0", None))

print("=== DeepSpeed ZeRO Summary ===")
for e, tr, vl, t, thr, mem in zip(
    epochs, train_loss, val_loss, epoch_time, samples_per_sec_global, peak_mem_gb
):
    print(
        f"Epoch {e}: "
        f"train_loss={tr:.4f}, val_loss={vl:.4f}, "
        f"epoch_time={t:.1f}s, global_throughput={thr:.2f} samples/s, "
        f"peak_mem_rank0={mem} GB"
    )

plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Train loss")
plt.plot(epochs, val_loss, marker="o", label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (cross-entropy)")
plt.title("BLIP-2 DeepSpeed ZeRO: Train vs Val loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
loss_plot_path = os.path.join(OUT_DIR, "loss_curves.png")
plt.savefig(loss_plot_path)
print(f"Saved loss curves to: {loss_plot_path}")

plt.figure()
plt.plot(epochs, samples_per_sec_global, marker="o", label="Global samples/s")
plt.xlabel("Epoch")
plt.ylabel("Samples per second (all GPUs)")
plt.title("DeepSpeed ZeRO: Global Throughput per Epoch")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
thr_plot_path = os.path.join(OUT_DIR, "throughput.png")
plt.savefig(thr_plot_path)
print(f"Saved throughput plot to: {thr_plot_path}")
