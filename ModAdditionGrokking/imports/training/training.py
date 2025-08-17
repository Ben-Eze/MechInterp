import matplotlib.pyplot as plt
import torch

from imports.plotting.plot_training import initialise_plot, update_plot


def train_model(
        model, optimiser, 
        N_epochs, eval_step, 
        batcher,
        live_update=False
        ):
    model.train()
    acc_train_history = []
    acc_test_history = []
    loss_train_history = []
    loss_test_history = []
    
    if live_update:
        fig, axes, lines = initialise_plot()

    epochs_completed = 0
    interrupted = False

    for epoch_i in range(N_epochs):
        try:
            X_train, y_train = batcher("train")
            logits_train, loss_train = model(X_train, y_train)
            optimiser.zero_grad()
            loss_train.backward()
            optimiser.step()
            epochs_completed = epoch_i + 1

            # Accuracy for training
            y_pred_train = torch.argmax(logits_train, dim=1)
            acc_train = (y_pred_train == y_train).float().mean().item()
            acc_train_history.append(acc_train)
            loss_train_history.append(loss_train.item())

            # Accuracy for test
            X_test, y_test = batcher("test")
            model.eval()
            with torch.no_grad():
                logits_test, loss_test = model(X_test, y_test)
                y_pred_test = torch.argmax(logits_test, dim=1)
                acc_test = (y_pred_test == y_test).float().mean().item()
                acc_test_history.append(acc_test)
                loss_test_history.append(loss_test.item())
            model.train()

            if live_update:
                update_plot(
                    acc_train_history, acc_test_history, loss_train_history, loss_test_history, lines
                )

            if not (epoch_i % eval_step):   # once every `eval_step` epochs
                # show test & train accuracy & loss
                print(f"{epoch_i=}  train loss: {loss_train.item():<.4f}, test loss: {loss_test.item():<.4f} | train acc: {acc_train:<.4f}, test acc: {acc_test:<.4f}")
        except KeyboardInterrupt:
            interrupted = True
            break

    if live_update:
        plt.ioff()
        plt.show()

    final_loss = loss_test.item()

    training_status = {
        "epochs_completed": epochs_completed,
        "interrupted": interrupted
    }

    # Output history data
    return {
        "model": model,
        "final_loss": final_loss,
        "training_status": training_status,
        "acc_train_history": acc_train_history,
        "acc_test_history": acc_test_history,
        "loss_train_history": loss_train_history,
        "loss_test_history": loss_test_history,
    }
