import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--exp', type=str, default='Unlearn', choices=["Unlearn", "Attack"])
    parser.add_argument('--method', type=str, default='MEGU')
    parser.add_argument('--target_model', type=str, default='GCN', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN", "SGC"])

    parser.add_argument('--inductive', type=str, default='normal', choices=['cluster-gcn', 'graphsaint', 'normal'])

    ########################## unlearning task parameters ######################
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics", "flickr", "ppi", "Photo", "Computers"])
    parser.add_argument('--unlearn_task', type=str, default='node', choices=['feature', "node", "edge"])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=2048)

    parser.add_argument('--unlearn_lr', type=float, default=0.05)
    parser.add_argument('--kappa', type=float, default=0.01)
    parser.add_argument('--alpha1', type=float, default=0.8)
    parser.add_argument('--alpha2', type=float, default=0.5)

    args = vars(parser.parse_args())

    return args
