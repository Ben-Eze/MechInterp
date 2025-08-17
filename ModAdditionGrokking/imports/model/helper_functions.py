import json
import os

import torch
import matplotlib.pyplot as plt


def save_model(model, final_loss, parent_path,
               acc_train_history, acc_test_history,
               loss_train_history, loss_test_history,
               hyperparameters, training_status,
               fig):
    # Ensure parent models directory exists
    models_dir = os.path.join(parent_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Find next unique code
    existing = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    codes = []
    for name in existing:
        try:
            codes.append(int(name.split("_")[0]))
        except ValueError:
            pass
    next_code = max(codes, default=-1) + 1
    code_str = f"{next_code:04d}"

    # Create directory for this model
    dir_name = f"{code_str}_loss={final_loss:.2f}"
    save_dir = os.path.join(models_dir, dir_name)
    os.makedirs(save_dir, exist_ok=False)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save history tensors
    torch.save(acc_train_history, os.path.join(save_dir, "acc_train_history.pt"))
    torch.save(acc_test_history, os.path.join(save_dir, "acc_test_history.pt"))
    torch.save(loss_train_history, os.path.join(save_dir, "loss_train_history.pt"))
    torch.save(loss_test_history, os.path.join(save_dir, "loss_test_history.pt"))

    # Save figure as PNG
    fig.savefig(os.path.join(save_dir, "plot.png"), dpi=300, bbox_inches="tight")

    # Save hyperparameters and training status
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump({
            "HYPERPARAMETERS": hyperparameters,
            "training_status": training_status
        }, f)

    print(f"Model and histories saved to `{save_dir}`")
