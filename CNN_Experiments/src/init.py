import torch
import models.CNN_MNIST     # required if using model even if shaded out!
from models.SmartSequential import SmartSequential


def get_model(MH, DEVICE="cpu"):
    if "LOAD" in MH and MH["LOAD"] is not None:
        state_dict = torch.load(MH["LOAD"])
        # get the Class
        ModelClass = SmartSequential.module_dict[state_dict["MODEL_CLASS"]]    
        model = ModelClass(state_dict["CONFIG"])
        model.load_state_dict(state_dict)
    else:
        ModelClass = SmartSequential.module_dict[MH["MODEL_CLASS"]]
        model = ModelClass(CONFIG=MH["CONFIG"])

    return model.to(DEVICE)


def get_optimiser(model, TH):
    if TH["OPTIMISER"] == "AdamW":
        optimiser = torch.optim.AdamW(
            params=model.parameters(), 
            lr=TH.get("LR", 0.01),
            weight_decay=TH.get("WEIGHT_DECAY", 0)
        )
    elif TH["OPTIMISER"] == "SGD":
        optimiser = torch.optim.SGD(
            params=model.parameters(), 
            lr=TH.get("LR", 0.01),
            weight_decay=TH.get("WEIGHT_DECAY", 0)
        )
    else:
        raise ValueError(f"Unknown optimiser: {TH['OPTIMISER']}")

    return optimiser


def get_scheduler(optimiser, TH):
    if TH["SCHEDULER"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimiser, 
            mode="min", 
            patience=TH.get("SCHEDULER_PATIENCE", 3), 
            factor=TH.get("SCHEDULER_FACTOR", 0.5)
        )
        step_scheduler = lambda scheduler, loss_test: scheduler.step(loss_test)
    elif TH["SCHEDULER"] == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimiser, 
            gamma=TH.get("SCHEDULER_GAMMA", 0.9)
        )
        step_scheduler = lambda scheduler, _: scheduler.step()
    else:
        raise ValueError(f"Unknown scheduler: {TH['SCHEDULER']}")
    
    return scheduler, step_scheduler


def get_loss_function(TH):
    if TH["LOSS_FUNCTION"] == "CrossEntropyLoss":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {TH['LOSS']}")

    return loss_function