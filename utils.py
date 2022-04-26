import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid,cifar_iid, cifar_noniid



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 0:
            # Sample IID user data from Mnist
            user_groups_train = cifar_iid(train_dataset, args.num_users)
            user_groups_val = cifar_iid(test_dataset, args.num_users)

        else:
            user_groups_train = cifar_noniid(train_dataset, args.num_users, val = False)
            user_groups_val =  cifar_noniid(test_dataset, args.num_users, val = True)

    elif args.dataset == 'mnist':

        data_dir = './data'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid == 0:
            # Sample IID user data from Mnist
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            user_groups_val = mnist_iid(test_dataset, args.num_users)
        else:

            user_groups_train = mnist_noniid(train_dataset, args.num_users)
            user_groups_val = mnist_noniid(test_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups_train, user_groups_val

