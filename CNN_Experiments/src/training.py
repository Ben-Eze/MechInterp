from src import utils
import torch
import warnings
import tqdm


def training_loop(model, optimiser, loss_function, scheduler, step_scheduler,
                  dataloader_train, dataloader_test,
                  N_EPOCHS, EVAL_INTERVAL, PRINT_PERFORMANCE=True):
    model.train()
    curr_performance = None
    training_complete = False

    try:
        for epoch_i in range(N_EPOCHS):
            for batch_i, (X_train, y_train) in tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                y_logits = model(X_train)

                # back propagation
                loss_train = loss_function(y_logits, y_train)
                optimiser.zero_grad()
                loss_train.backward()
                optimiser.step()
            
            performance = eval_performance(
                model, optimiser, loss_function, scheduler, step_scheduler,
                y_train, dataloader_test,
                y_logits, loss_train,
                epoch_i, EVAL_INTERVAL)
            
            if performance is not None:
                curr_performance = performance
                
                if PRINT_PERFORMANCE:
                    print_performance(performance)
        
        training_complete = True
        
        # get final model performance if not up-to-date
        if performance is None:
            curr_performance = eval_performance(
                    model, optimiser, loss_function, scheduler, step_scheduler,
                    y_train, dataloader_test,
                    y_logits, loss_train,
                    epoch_i, EVAL_INTERVAL=1)
    except Exception as e:
        warnings.warn(e.args[0])
    except:
        pass

    print(f"Training Complete: {training_complete}\n")
    
    return model, curr_performance, training_complete


def eval_performance(model, optimiser, loss_function, scheduler, step_scheduler,
                     y_train, dataloader_test,
                     y_logits, loss_train,
                     epoch_i, EVAL_INTERVAL):
    if epoch_i % EVAL_INTERVAL:
        return None
    
    # Evaluate performance
    y_pred_prob = y_logits.softmax(dim=1)
    y_pred = y_pred_prob.argmax(dim=1)
    accuracy_train = utils.accuracy_function(pred=y_pred, truth=y_train)
    
    loss_test, accuracy_test = get_test_performance(
        model, loss_function, dataloader_test, 
    )

    step_scheduler(scheduler, loss_test)

    return {
        "epoch_i": epoch_i,
        "lr": optimiser.param_groups[0]['lr'], 
        "loss_train": loss_train,
        "loss_test": loss_test,
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test
    }

def print_performance(p):
    print(f"Epoch {p['epoch_i']}:\n"
          f"LR = {p['lr']}\n"
          f"Training loss = {p['loss_train']}\n"
          f"Test loss = {p['loss_test']}\n"
          f"Training acc = {p['accuracy_train']}\n"
          f"Test acc = {p['accuracy_test']}\n"
        )


def get_test_performance(model, loss_func, dataloader_test):
    model.eval()    # sets dropout & batch norm for evaluation
    N_test = len(dataloader_test.dataset.data)
    accuracy = 0
    
    for (X_test, y_test) in dataloader_test:
        with torch.inference_mode():    # disables gradient
            y_logits = model(X_test)
        
        loss = loss_func(y_logits, y_test)
        
        y_pred_prob = y_logits.softmax(dim=1)
        y_pred = y_pred_prob.argmax(dim=1)
        accuracy += utils.accuracy_function(y_pred, y_test) * len(X_test)/N_test

    model.train()
    
    return loss, accuracy