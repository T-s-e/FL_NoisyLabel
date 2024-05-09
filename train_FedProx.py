import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from collections import Counter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils.options import args_parser
from utils.local_training import LocalUpdate, globaltest
from utils.FedAvg import FedAvg
from utils.utils import add_attribute_noise, set_seed, set_output_files

from dataset.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
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
    logging.info(f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(f"valid: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_staining = np.array(dataset_train.staining)

    # specify the noise rate for each staining
    staining_noise_rate = [0.4, 0.4, 0.3, 0.2, 0.1]
    # staining_noise_rate = [0.7, 0.7, 0.5, 0.3, 0.1]
    logging.info(f"staining_noise_rate: {staining_noise_rate}")

    y_train_noisy, gamma_s, real_noise_level = add_attribute_noise(args, y_train, y_staining, dict_users, staining_noise_rate)
    dataset_train.clean_targets = np.array(dataset_train.targets)
    dataset_train.targets = y_train_noisy

    # --------------------- Build Models ---------------------------
    netglob = build_model(args)
    # net_local = build_model(args)

    user_id = list(range(args.n_clients))
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(dataset_train), dict_users[id]))

    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------beging training---------------------")
    best_performance = 0.
    BACC = []
    for rnd in range(args.rounds):
        w_locals, loss_locals = [], []
        for idx in user_id:  # training over the subset
            local = trainer_locals[idx]
            w_local, loss_local = local.train_fedprox(
                net=copy.deepcopy(netglob).to(args.device), writer=writer)
            # store every updated model
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in user_id]
        print(dict_len)
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        pred = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        acc = accuracy_score(dataset_test.targets, pred)
        bacc = balanced_accuracy_score(dataset_test.targets, pred)
        cm = confusion_matrix(dataset_test.targets, pred)
        logging.info("******** round: %d, acc: %.4f, bacc: %.4f ********"  % (rnd, acc, bacc))
        logging.info(cm)
        writer.add_scalar(f'test/acc', acc, rnd)
        writer.add_scalar(f'test/bacc', bacc, rnd)
        BACC.append(bacc)

        # save model
        if bacc > best_performance:
            best_performance = bacc
            # torch.save(netglob.state_dict(),  models_dir +
            #            f'/best_model_{rnd}_{best_performance}.pth')
            # torch.save(netglob.state_dict(),  models_dir+'/best_model.pth')
        logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')

        logging.info('\n')
        
    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())

    torch.cuda.empty_cache()
