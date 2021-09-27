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

    indexClass={}

    for i,value in model.class_to_idx.items():
        indexClass[value]=i
    probability, classes = utils.ImagePrediction(args.input, model,args.top_k,device,indexClass)
      
    if(args.category_names):
        cat_to_name=[]
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            names = [cat_to_name[c] for c in classes]

    print('Classes: {}','probabilities(%): {}','Top Classes: {}'.format(names,[float(round(p * 100.0, 2)) for p in probs],names[0]))




def main():
    parser = argparse.ArgumentParser(description='Flowers Classifcation Trainer')
    parser.add_argument('--gpu', type=bool, default=False, help='Enable/Disable GPU')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: densenet121, vgg16]')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='path to en existance  checkpoint',required=True)
    parser.add_argument('--top_k', type=int, default=5, help='top k classes for the input')
    parser.add_argument('--category_names', type=str, help='json path file of categories names of flowers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for  fc layer')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--save_dir' , type=str, default='./', help='checkpoint directory path')
    parser.add_argument('--input', type=str, help='path for image to predict')

    args = parser.parse_args()
    LoadPredict(args)
    print("Successfully completed\n")
