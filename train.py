from Modules import *
import utils
import network


def Tr_function(args):
    if(args.gpu):
        if torch.cuda.is_available():
            device = 'gpu'
            print('cuda available')
        else:
            device = 'cpu'
            print('Cuda not available, train with cpu')
    else:
        device = 'cpu'