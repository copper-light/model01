import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNN

def dataset(path, batch_size):
    train_data = datasets.MNIST(path, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])) # 학습 데이터
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    test_data = datasets.MNIST(path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])) # 테스트 데이터
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def train(model, loader, criterion, optimizer, epoch):
    model.train()  # 학습을 위함    
    for epoch in range(epoch):
        for index, (data, target) in enumerate(loader):
            optimizer.zero_grad()  # 기울기 초기화
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # 역전파
            optimizer.step()
        
            if index % 100 == 0:
                print("loss of {} epoch, {}/{} index : {}".format(epoch, index, len(train_loader), loss.item()))


def test(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('\{"loss": {:.4f}, "correct": {}/{}, "accuracy":{:.0f}\}'.format( test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



if __name__ == "__main__":
    lr = 0.01
    epoch = 2
    batch_size = 64
    data_path = '../data/'

    print("start train cnn")
    cnn = CNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=lr)
    train_loader, test_loader = dataset(data_path, batch_size)
    train(cnn, train_loader, criterion, optimizer, epoch)
    test(cnn, test_loader, criterion)
    