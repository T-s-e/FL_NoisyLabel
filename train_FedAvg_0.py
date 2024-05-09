import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.local_training import LocalUpdate_FedAvg, globaltest
from utils.FedAvg import FedAvg, DaAgg
from utils.utils import add_noise, add_attribute_noise, set_seed, set_output_files, get_output, get_current_consistency_weight

from dataset.dataset import get_dataset
from model.build_model import build_model
from model.Nets import CNN
from model.FedAvg import FedAvg, test_img

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients     # default: 20
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(ROOT_DIR, args)

    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    logging.info(
        f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_staining = np.array(dataset_train.staining)

    # specify the noise rate for each staining
    staining_noise_rate = [0.4, 0.4, 0.3, 0.2, 0.1]
    # staining_noise_rate = [0.7, 0.7, 0.5, 0.3, 0.1]
    logging.info(f"staining_noise_rate: {staining_noise_rate}")

    y_train_noisy, gamma_s, real_noise_level = add_attribute_noise(
        args, y_train, y_staining, dict_users, staining_noise_rate)
    dataset_train.targets = y_train_noisy

    # --------------------- Build Models ---------------------------

    img_size = dataset_train[0][0].shape

    net_glob = CNN(args=args).to(args.device)
    logging.info(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # --------------------- Training ---------------------------
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = 1.0
    val_acc_list, net_list = [], []

    w_locals = [w_glob for _ in range(args.n_clients)]
    for iter in range(args.rounds):
        loss_locals = []

        # All clients update
        idxs_users = np.array(range(args.n_clients))

        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            # print client loss
            logging.info(f'Round {iter}, User {idx}, loss: {loss:.3f}')

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if best_loss > loss_avg:
            best_loss = loss_avg
        logging.info('Round {:3d}, Average loss {:.3f}, Best loss {:.3f}'.format(iter, loss_avg, best_loss))
        loss_train.append(loss_avg)


    # Save the model
    torch.save(net_glob.state_dict(), models_dir + '/final_model.pth')
    logging.info(f'Model saved to {models_dir}/final_model.pth')

    model_path = models_dir + '/final_model.pth'
    net_glob.load_state_dict(torch.load(model_path))
    logging.info(f'Model loaded from {model_path}')




    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train, label='train')
    plt.title('Camelyon17')
    plt.ylabel('loss')
    plt.xlabel('round')
    plt.savefig(f'{models_dir}/loss_0.png')
    plt.close()

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    logging.info("Training accuracy: {:.2f}".format(acc_train))
    logging.info("Testing accuracy: {:.2f}".format(acc_test))

    torch.cuda.empty_cache()

