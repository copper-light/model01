import torch
import torch.optim as optim
from torchvision import datasets, transforms
from cnn import CNN
import logging


def dataset(path, batch_size):
    train_data = datasets.MNIST(path, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])) 
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    test_data = datasets.MNIST(path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])) 
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def train(model, loader, criterion, optimizer, epoch):
    model.train() 
    for epoch in range(epoch):
        for index, (data, target) in enumerate(loader):
            optimizer.zero_grad()  
            output = model(data)
            loss = criterion(output, target)
            loss.backward() 
            optimizer.step()
        
            if index % 100 == 0:
                print("loss of {} epoch, {}/{} index : {}".format(epoch, index, len(loader), loss.item()))


def test(model, loader, criterion):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('{{"loss": {:.4f}, "correct": {}, "total": {}, "accuracy":{:.4f}}}'.format(loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
        print('loss={:.4f}'.format(loss))
        print('accuracy={:.4f}'.format(correct / len(loader.dataset)))


def run(lr, epoch, batch_size, data_path):
    print("start train cnn")
    cnn = CNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=lr)
    train_loader, test_loader = dataset(data_path, batch_size)
    train(cnn, train_loader, criterion, optimizer, epoch)
    test(cnn, test_loader, criterion)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--data-path', type=str, default='./data')
    # parser.add_argument('--log_', type=str, default='./data')
    
    args = parser.parse_args()

    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    data_path = args.data_path

    run(lr, epoch, batch_size, data_path)
    