import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_transforms():
    transform_basic = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_augmented = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    return transform_basic, transform_augmented

def get_datasets():
    transform_basic, transform_augmented = get_transforms()
    train_set_basic = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_basic)
    train_set_aug = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_augmented)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_basic)
    return train_set_basic, train_set_aug, test_set

def get_emnist_datasets(split="balanced"):
    transform_basic, transform_augmented = get_transforms()
    train_set_basic = torchvision.datasets.EMNIST(root='./data', split=split, train=True, download=True, transform=transform_basic)
    train_set_aug = torchvision.datasets.EMNIST(root='./data', split=split, train=True, download=True, transform=transform_augmented)
    test_set = torchvision.datasets.EMNIST(root='./data', split=split, train=False, download=True, transform=transform_basic)
    return train_set_basic, train_set_aug, test_set

def get_loader(augmented=False, batch_size=64, num_workers=4, pin_memory=True, persistent_workers=True):
    train_set_basic, train_set_aug, _ = get_datasets()
    dataset = train_set_aug if augmented else train_set_basic
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

def get_emnist_loader(split="balanced", augmented=False, batch_size=64, num_workers=4, pin_memory=True, persistent_workers=True):
    train_set_basic, train_set_aug, _ = get_emnist_datasets(split=split)
    dataset = train_set_aug if augmented else train_set_basic
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

def get_test_loader(batch_size=1000, num_workers=4, pin_memory=True, persistent_workers=True):
    _, _, test_set = get_datasets()
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

def get_emnist_test_loader(split="balanced", batch_size=1000, num_workers=4, pin_memory=True, persistent_workers=True):
    _, _, test_set = get_emnist_datasets(split=split)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

def get_full_train_set(augmented=False):
    transform_basic, transform_augmented = get_transforms()
    transform = transform_augmented if augmented else transform_basic
    return torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def get_full_emnist_train_set(split="balanced", augmented=False):
    transform_basic, transform_augmented = get_transforms()
    transform = transform_augmented if augmented else transform_basic
    return torchvision.datasets.EMNIST(root='./data', split=split, train=True, download=True, transform=transform)
