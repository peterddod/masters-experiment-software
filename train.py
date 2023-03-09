"""
Training functions for activation pattern experiment.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import torch


def run_epoch(model, optimiser, loss, train_datapoint, test_data, target, epoch, batch_idx, file):
    loss = train_step(model, optimiser, loss, train_datapoint, target)
    accuracy = 0  # evaluate_epoch(model, test_data)

    file( ",".join([str(epoch),str(batch_idx),str(loss),str(accuracy)]))

    if batch_idx % 10 == 0:      
        print(f'Epoch {epoch}, Batch: {batch_idx}, Loss: {loss}, Accuracy: {accuracy}')


def train_step(model, optimiser, loss, data, target):
    model.train()
    optimiser.zero_grad()
    output = model(data)
    output = loss(output, target)
    output.backward()
    optimiser.step()

    return output.item()

def evaluate_epoch(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)

            total += data.size(0)
            correct += torch.sum(output.argmax(dim=1) == target)

    accuracy = (correct/total).item()

    return accuracy

def save_state_dict(state_dict, path):
    torch.save(state_dict, path)

def load_model(model_cls, path):
    model = model_cls()
    model.load_state_dict(torch.load(path))
    return model
