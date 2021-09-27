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

    
    # TODO:task1 DEFINE DATA TRANSFORM, DEFINING DATASETS 

    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                    ])
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                       ])
