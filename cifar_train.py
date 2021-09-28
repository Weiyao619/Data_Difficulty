import random
from utils import *
from parser_args import *
from main_workers import *
from dataset_maker import *


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)
    #main_worker_pbar(args.gpu, args)
    #main_worker_l(args.gpu, args)
    #Err = torch.load('Err.pt')
    #index = Sorting(Err)
    #torch.save(IndexMatch(index,args),'trainset_sorted_Error.pt')
    #Loss = torch.load('L.pt')
    #index = Sorting(Loss)
    #torch.save(IndexMatch(index,args),'trainset_sorted_Loss.pt')
    #torch.save(NoiseProductor(args),'trainset_noisy80_cifar10.pt')
   


if __name__ == '__main__':
    main()