from prefect import task, flow
from torchvision import datasets, transforms

#Define downloading the data as a task
@task
def download_data():
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform= \
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform= \
                                               transforms.Compose([transforms.ToTensor()]))  
    return train_set, test_set

