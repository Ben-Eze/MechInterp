import torchvision
from torch.utils.data import DataLoader


def load_data(dataset, BATCH_SIZE_TRAIN=None, BATCH_SIZE_TEST=None):
    dset2func = {"MNIST": torchvision.datasets.MNIST,
                 "FashionMNIST": torchvision.datasets.FashionMNIST}
    try:
        dataset_function = dset2func[dataset]
    except KeyError:
        raise KeyError(f"dataset '{dataset}' not recognised. Must be one of the following: {dset2func.keys()}")
    
    data_train = dataset_function(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None
    )

    data_test = dataset_function(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None
    )
    
    if BATCH_SIZE_TRAIN is None:
        BATCH_SIZE_TRAIN = len(data_train)

    if BATCH_SIZE_TEST is None:
        BATCH_SIZE_TEST = len(data_test)
    
    dataloader_train = DataLoader(
        dataset=data_train,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True
    )

    dataloader_test = DataLoader(
        dataset=data_test,
        batch_size=BATCH_SIZE_TEST
    )

    N_images_test = len(dataloader_test.dataset)
    N_batches_test = len(dataloader_test)
    N_images_train = len(dataloader_train.dataset)
    N_batches_train = len(dataloader_train)

    return dataloader_train, dataloader_test, \
           N_images_test, N_batches_test, \
           N_images_train, N_batches_train