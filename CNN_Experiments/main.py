import torch
from src import data
from src import training
from src import hyperparams
from src import save_load
from src import init


def main(config):
    # HYPERPARAMETERS:
    #   DH - data, MH - model, TH - training
    DH, MH, TH = hyperparams.read(config)

    # PYTORCH 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed=TH["SEED"])

    # DATA
    dataloader_train, dataloader_test, *_ = \
        data.load_data(dataset=DH["DATASET"], BATCH_SIZE_TRAIN=2048)
    
    # MODEL, OPTIMISER, SCHEDULER, LOSS
    model = init.get_model(MH, DEVICE)
    optimiser = init.get_optimiser(model, TH)
    scheduler, step_scheduler = init.get_scheduler(optimiser, TH)
    loss_function = init.get_loss_function(TH)

    # TRAIN
    model, curr_performance, training_complete = training.training_loop(
        model=model, optimiser=optimiser, loss_function=loss_function, 
        scheduler=scheduler, step_scheduler=step_scheduler,
        dataloader_train=dataloader_train, dataloader_test=dataloader_test,
        N_EPOCHS=TH["N_EPOCHS"], EVAL_INTERVAL=TH["EVAL_INTERVAL"]
    )

    # SAVE
    save_load.save(model, 
                file_path=MH["SAVE_FILE"], 
                loss=curr_performance["loss_test"], 
                acc=curr_performance["accuracy_test"], 
                )


if __name__ == "__main__":
    main("configs/CNN.json")