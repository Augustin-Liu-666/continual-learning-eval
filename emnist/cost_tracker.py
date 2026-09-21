"""
cost_tracker.py
===============
Tracks the computational cost of continual-learning methods, including:
  - training time per task
  - peak memory use (RAM/GPU)
  - additional storage (replay buffers or parameter snapshots)
  - approximate per-step computation

Example usage:
    from emnist.cost_tracker import CostTracker
    tracker = CostTracker(method_name="ER")
    tracker.start_task(task_id=0)
    ... training ...
    tracker.end_task(task_id=0)
    tracker.save_report("emnist/reports/cost_er.csv")
"""

import time
import os
import csv
import tracemalloc
import torch


class CostTracker:
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.records = []           # One record per task.
        self._task_start_time = None
        self._wall_start = None

    # ------------------------------------------------------------------
    # Task-level timing
    # ------------------------------------------------------------------
    def start_task(self, task_id: int):
        """Start timing and memory tracking for a task."""
        self._task_id = task_id
        self._task_start_time = time.perf_counter()
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_task(self, task_id: int,
                 extra_storage_bytes: int = 0,
                 extra_storage_label: str = ""):
        """
        Finish tracking a task.

        ``extra_storage_bytes`` records method-specific storage, such as an ER
        replay buffer or the parameter snapshots and importance weights used by
        MAS and OGD.
        """
        elapsed = time.perf_counter() - self._task_start_time

        # Peak RAM usage
        current, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak GPU usage, when available
        peak_gpu = 0
        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated()

        self.records.append({
            "method":             self.method_name,
            "task_id":            task_id,
            "train_time_s":       round(elapsed, 3),
            "peak_ram_mb":        round(peak_ram / 1024 / 1024, 2),
            "peak_gpu_mb":        round(peak_gpu / 1024 / 1024, 2),
            "extra_storage_mb":   round(extra_storage_bytes / 1024 / 1024, 4),
            "extra_storage_label": extra_storage_label,
        })
        print(f"  [CostTracker] Task {task_id} | "
              f"Time: {elapsed:.1f}s | "
              f"RAM peak: {peak_ram/1024/1024:.1f} MB | "
              f"Extra storage: {extra_storage_bytes/1024/1024:.2f} MB ({extra_storage_label})")

    # ------------------------------------------------------------------
    # Helpers for method-specific storage
    # ------------------------------------------------------------------
    @staticmethod
    def er_buffer_bytes(memory: list) -> int:
        """Estimate the number of bytes used by an ER replay buffer."""
        if not memory:
            return 0
        x0, _ = memory[0]
        bytes_per_sample = x0.element_size() * x0.nelement() + 4  # +4 for label int32
        return len(memory) * bytes_per_sample

    @staticmethod
    def params_bytes(model: torch.nn.Module) -> int:
        """Return the number of bytes in one trainable parameter snapshot."""
        return sum(p.nelement() * p.element_size()
                   for p in model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Report output
    # ------------------------------------------------------------------
    def save_report(self, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if not self.records:
            print("[CostTracker] No records available; skipping report output.")
            return
        fieldnames = list(self.records[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[CostTracker] Cost report saved to: {output_path}")

    def print_summary(self):
        """Print a task-level cost summary."""
        print(f"\n{'='*60}")
        print(f"  Cost Summary — Method: {self.method_name}")
        print(f"{'='*60}")
        print(f"{'Task':<6} {'Time(s)':<10} {'RAM(MB)':<12} {'GPU(MB)':<12} {'ExtraStore(MB)':<16} {'Label'}")
        print(f"{'-'*60}")
        total_time = 0
        for r in self.records:
            print(f"  T{r['task_id']:<4} {r['train_time_s']:<10} {r['peak_ram_mb']:<12} "
                  f"{r['peak_gpu_mb']:<12} {r['extra_storage_mb']:<16} {r['extra_storage_label']}")
            total_time += r['train_time_s']
        print(f"{'-'*60}")
        print(f"  Total training time: {total_time:.1f}s")
        print(f"{'='*60}\n")
