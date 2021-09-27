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