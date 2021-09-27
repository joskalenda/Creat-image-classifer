from Modules import*
import utils
import network


def LoadPredict(args):
 
    model=network.loadCheckpoint(args.checkpoint_path)

    if(in_arg.gpu):
        if torch.cuda.is_available():
            device = 'gpu'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    model.to(device)