from Modules import *
import utils




def creat_classifier(args):
    print('Select Archictecture network\n')
    if(args.arch=='vgg16'):
        model=models.vgg16(pretrained=True)
        Feature = model.classifier[0].in_features
        output_size = 102

    elif(args.arch == 'densenet121'):
        model=models.densenet121(pretrained= True)
        Feature = model.classifier.in_features
        output_size = 102

    else:
        raise Exception('Arch not compatible\n')

    print('creat pretrained classifier network\n')

    for parm in model.parameters():
        parm.requires_grad=False;

    # Features = model.classifier[0].in_features
    Classifier= nn.Sequential(OrderedDict([('dropout1', nn.Dropout(p=0.3)),
                                            ('fc1', nn.Linear(Feature, args.hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout2', nn.Dropout(p=0.3)),
                                            ('fc2', nn.Linear(args.hidden_units, output_size)),  
                                            ('relu2', nn.ReLU()),
                                            ('output', nn.LogSoftmax(dim = 1)),
                                        ]))
    model.classifier= Classifier
    return model
    model.class_to_idx = train_dataset.class_to_idx
    criterion = nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(),lr=args.lr)



def Model_function(args,model,optimizer,criterion,trainloader,valloader,epochs,device):
    
    print("Starting the Traing and calculate the time\n")
    start_time = time()
#     args = Arg_function()
    if(args.gpu):
        if torch.cuda.is_available():
            device = 'gpu'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
      
    model.to(device)
    accuracy = 0
    model.optim_state_dict = optimizer.stat

