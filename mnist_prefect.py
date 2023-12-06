from prefect import task, flow
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#Define downloading the data as a task
@task
def download_data():
    train_set = datasets.FashionMNIST("./data", download=True, transform= \
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = datasets.FashionMNIST("./data", download=True, train=False, transform= \
                                               transforms.Compose([transforms.ToTensor()]))  
    return train_set, test_set

#Loading data task
@task
def load_data(train_set, test_set):
    train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100)
    return train_loader, test_loader

#test visualization task
@task
def vis_test(train_set):
    image, label = next(iter(train_set))
    plt.imshow(image.squeeze(), cmap="gray")
    print(label)


#Create a work flow
@flow
def first_flow():
    #Tasks
    print('Flow')
    train_set, test_set = download_data()
    train_loader, test_laoder = load_data(train_set, test_set)
    vis_test(train_set)
