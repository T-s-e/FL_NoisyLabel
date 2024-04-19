import numpy as np
import torch
import os
import torchvision.transforms as transforms
import pandas as pd

from .all_datasets import isic2019, ICH, Camelyon17

from utils.sampling import iid_sampling, non_iid_dirichlet_sampling

DATA_DIR = '/home/xkx/'

def get_dataset(args):
    if args.dataset == "isic2019":
        root = "your path"
        args.n_classes = 8

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = isic2019(root, "train", train_transform)
        test_dataset = isic2019(root, "test", val_transform)


    elif args.dataset == "ICH":
        root = "your path"
        args.n_classes = 5

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = ICH(root, "train", train_transform)
        test_dataset = ICH(root, "test", val_transform)


    elif args.dataset == "Camelyon17":
        root = DATA_DIR + "Camelyon17"
        args.n_classes = 2

        normalize = transforms.Normalize([0.702, 0.546, 0.696],
                                         [0.238, 0.282, 0.224])
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = Camelyon17(root, "train", train_transform)
        test_dataset = Camelyon17(root, "test", val_transform)

        # dict_users_path = os.path.join(root, "train_dict_user.npy")
        # dict_users = np.load(dict_users_path, allow_pickle=True).item()


    else:
        exit("Error: unrecognized dataset")

    n_train = len(train_dataset)
    y_train = np.array(train_dataset.targets)
    assert n_train == len(y_train)

    y_staining = np.array(train_dataset.staining)

    if args.iid:
        dict_users = iid_sampling(n_train, args.n_clients, args.seed)
    else:
        # dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, args.non_iid_prob_class, args.n_clients,
        #                                         seed=100, alpha_dirichlet=args.alpha_dirichlet)
        dict_users = non_iid_dirichlet_sampling(y_staining, train_dataset.s_classes, args.non_iid_prob_class, args.n_clients,
                                                seed=100, alpha_dirichlet=args.alpha_dirichlet)
    dict_users = dict(sorted(dict_users.items()))
    np.save(os.path.join(root, "train_dict_user.npy"), dict_users)

    # check
    assert len(dict_users.keys()) == args.n_clients
    items = []
    for key in dict_users.keys():
        items += list(dict_users[key])
    assert len(items) == len(set(items)) == len(y_train)

    print("### Datasets are ready ###")
    
    return train_dataset, test_dataset, dict_users


def class_to_onehot(csv_file, save_path):
    df = pd.read_csv(csv_file)

    one_hot = pd.get_dummies(df['class'], prefix='class').astype(int)
    df = df.join(one_hot)
    df.drop('class', axis=1, inplace=True)

    # print(df.head())

    df.to_csv(save_path, index=False)
