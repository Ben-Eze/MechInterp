import pathlib

import torch

from imports.model.ModAdditionModel import ModAdditionModel
from imports.misc.helper_functions import save_model
from imports.plotting.plot_training import plot_final_histories
from imports.training.ModAdditionBatcher import ModAdditionBatcher
import imports.training.training as training


def train(HP):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent_path = pathlib.Path(__file__).parent.resolve()
    if HP["seed"] is not None:
        torch.manual_seed(HP["seed"])

    model = ModAdditionModel(p=HP["p"], d=HP["d"], N_heads=HP["N_heads"], d_head=HP["d_head"], n=HP["n"]).to(device)

    # TODO: delete
    # model.load_state_dict(torch.load(f"{parent_path}/models/gpt_loss=", map_location=device))

    optimiser = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["lambda"])

    batcher = ModAdditionBatcher(device=device, p=HP["p"], frac_train=HP["frac_train"], n_sample_default=int(HP["frac_train"]*HP["p"]**2))

    out = training.train_model(
        model, optimiser, 
        HP["N_epochs"], eval_step=HP["eval_step"], 
        batcher=batcher,
        live_update=False
    )

    fig = plot_final_histories(out["acc_train_history"], out["acc_test_history"], 
                               out["loss_train_history"], out["loss_test_history"])

    # save_model(model, final_loss, parent_path)
    save_model(model=out["model"], final_loss=out["final_loss"], parent_path=parent_path,
               acc_train_history=out["acc_train_history"], acc_test_history=out["acc_test_history"],
               loss_train_history=out["loss_train_history"], loss_test_history=out["loss_test_history"],
               hyperparameters=HYPERPARAMETERS, training_status=out["training_status"],
               fig=fig)


if __name__ == "__main__":
    HYPERPARAMETERS = {
        "description": "Full run",
        "lr": 1e-3,
        "N_epochs": 40000,
        "p": 113,
        "d": 128,
        "N_heads": 4,
        "d_head": 32,
        "n": 512,
        "lambda": 5,
        "frac_train": 0.3,
        "eval_step": 10,
        "seed": 123
    }
    train(HYPERPARAMETERS)