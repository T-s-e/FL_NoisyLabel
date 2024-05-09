import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='FedAvg', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='Camelyon17', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--pretrained', type=int,  default=0)
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")


    # for FL
    parser.add_argument('--n_clients', type=int,  default=20,
                        help='number of users') 
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=5, help='local epoch')
    parser.add_argument('--rounds', type=int,  default=40, help='rounds')

    # FedCorr
    parser.add_argument('--iteration1', type=int,  default=5, help='iteration1')
    parser.add_argument('--rounds1', type=int, default=10, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=25, help="rounds of training in usual training stage")
    # parser.add_argument('--frac1', type=float, default=0.1, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.2, help="fration of selected clients in fine-tuning and usual training stage")
    parser.add_argument('--relabel_ratio', type=float, default=0.5,
                        help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')
    parser.add_argument('--mixup', action='store_true')

    # FedProx
    parser.add_argument('--prox', type=float,  default=0.01, help='proximal term constant')



    parser.add_argument('--s1', type=int,  default=5, help='stage 1 rounds')
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a', type=float,  default=0.8, help='a')
    parser.add_argument('--warm', type=int,  default=1)

    # noise
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default="instance", help="type of noise")

    args = parser.parse_args()
    return args
