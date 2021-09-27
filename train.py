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

     
    # TODO: Load the datasets with ImageFolder
    print('Loading Train data....')
    train_dataset =datasets.ImageFolder(train_dir,transform=train_transform) 
    test_dataset =datasets.ImageFolder(test_dir,transform=test_transform) 
    val_dataset =datasets.ImageFolder(valid_dir,transform=val_transform) 
 
     # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    valloader= torch.utils.data.DataLoader(val_dataset,batch_size=32)
    testloader= torch.utils.data.DataLoader(test_dataset,batch_size=32);

    print('Preparing the model...\n')
    model = network.creat_classifier(args) 
    print('model ready\n')
    print('start training...\n')

    optimizer= optim.Adam(model.classifier.parameters(),lr=args.lr)
    model.epoch = args.epochs
    criterion = nn.NLLLoss()


    model = network.Model_function(args,model, optimizer, criterion, trainloader, valloader, args.epochs, args.save_dir)
    print('training completed\n')
    model.epoch = args.epochs
    model.class_to_idx = train_dataset.class_to_idx

