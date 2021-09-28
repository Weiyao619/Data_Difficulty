import torch
from utils import *
from parser_args import *
from main_workers import *
from dataset_maker import *

if __name__ == '__main__':
    args = parser.parse_args()
    Err = torch.load('Err.pt')
    L1 = Sorting(Err).cpu().numpy()
    Loss = torch.load('L.pt')
    L2 = Sorting(Loss).cpu().numpy()
    print(Sorting_index_comparaison(L1,L2,args))