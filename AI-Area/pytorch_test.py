import torch

from torch import nn
from torch.utils.data import DataLoader # Dataloader wraps an iterable around dataset
from torchvision import datasets # dataset stores the samples and their corresponding labels
from torchvision.transforms import v2

from NN_models import NueralNetwork

batch_size = 64

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = NueralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
# for the NN



training_data = datasets.FashionMNIST(
    root = "data" ,
    train = True,
    download = True,
    transform= v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
);

test_data = datasets.FashionMNIST(
    root= "data",
    train=False,
    download=True,
    transform= v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)])
)




def train(dataloader, model , loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch , (X,y) in enumerate(dataloader):
        X, y = X.to(device) , y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:5>d}]/{size:>5d}")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0 , 0
    with torch.no_grad():
        for X, y in dataloader:
            X , y = X.to(device) , y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    epochs = 5
    train_dataloader = DataLoader(training_data, batch_size= batch_size)

    test_dataloader = DataLoader(test_data, batch_size= batch_size)

    # for X , y  in test_dataloader:
    #     print(f"Shape of X [N , C , H ,W] : {X.shape}")
    #     print(f"Shape of y: {y.shape}{y.dtype}")
    #     break

    for t in range(epochs):
        print(f"Epoch {t+1} \n-----------------------------")
        train(train_dataloader, model,loss_fn,optimizer)

        test(test_dataloader,model,loss_fn)

    X = torch.rand(1,28,28 , device= device)
    
    print("Cleared")
    


if __name__ == "__main__":
    main()


