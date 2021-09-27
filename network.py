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
    model.optim_state_dict = optimizer.state_dict()

    for e in range(epochs):

        running_loss=0;
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad();
            output=model(images)
            loss=criterion(output,labels)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                val_loss=0;
                save_accuracy= -5;#save best accuracy
               
                for images , labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    output=model(images)
                    loss=criterion(output,labels)
                    val_loss+=loss.item()
                    output_Exp=torch.exp(output)
                    top_p,top_c = output_Exp.topk(1,dim=1)
                    equals= top_c ==labels.view(*top_c.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                model.optim_state_dict=optimizer.state_dict()
                
                if accuracy>save_accuracy:
                    accuracy=save_accuracy
                    model.save_accuracy=accuracy
                    model.epoch=e
                    
                    saveCheckPoint(model,'checkpoint.ph')
    
                train_lss = running_loss/len(trainloader)
                val_lss = val_loss/len(valloader)
                test_lss  =  save_accuracy/len(valloader)
                print(f"Epoch {e+1}/{epochs}.. ",
                  "Train loss: {}","Valloss: {}","valac: {}".format(train_lss,val_lss,test_lss))
                 
        model.train()
    return model
    print("Training done with success\n")

    end_time = time()
    tot_time = end_time - start_time
    tot_time = strftime('%H:%M:%S', localtime(tot_time))
    print("\n** Total Elapsed Training Runtime: ", tot_time)
     




def evaluate(model,criterion,testloader,device):
    with torch.no_grad():
            model.eval()
            val_loss=0;
            accuracy=0;
            for images , labels in loader:
                images, labels = images.to(device), labels.to(device)
                output=model(images)
                loss=def_criterion(output,labels)
                val_loss+=loss.item()
                output_Exp=torch.exp(output)
                top_p,top_c = output_Exp.topk(1,dim=1)
                equals= top_c ==labels.view(*top_c.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"test  loss: {val_loss/len(loader):.3f}.. "
                  f"test  accuracy: {accuracy/len(loader):.3f}")
    model.train()





def saveCheckPoint(model,args):
    if(args.arch=='vgg16'):
        model=models.vgg16(pretrained=True)
        Feature = model.classifier[0].in_features
    elif(args.arch == 'densenet121'):
        model=models.densenet121(pretrained= True)
        Feature = model.classifier.in_features
    else:
        raise Exception('Arch not compatible\n')
    
        
    checkpoint = {
               'state_dict': model.state_dict(),
               'epoch': model.epoch,
               'optimizer_state':model.optim_state_dict,
               'class_to_idx': model.class_to_idx,
               'output_size': 102,
               'fearure':Feature,
               'hidden_units':args.hidden_units,
                'accuracy':model.accuracy
             }
      # Create a folder to save checkpoint if not already existed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint,save_dir +'checkpoint.pth')
    print('Model saved successfully!')




       
def loadCheckpoint(checkpointPath):
    checkpoint =torch.load(checkpointPath)
    model.classifier= nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc1', nn.Linear(checkpoint['feature'],checkpoint['hidden_units'] )),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
                              ('relu2', nn.ReLU()),
                              ('dropout3', nn.Dropout(p=0.5)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    
                                
    model.class_to_idx=checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.epoch=checkpoint['epoch']
    model.optimizer_state=checkpoint['optimizer_state']
    print("Loading checkpoint done with sucess\n")
    return model




def Arg_function():
    parser = argparse.ArgumentParser(description='Flowers Classifcation Trainer')
    parser.add_argument('--input', type=str, help='path for image to predict')
    parser.add_argument('--gpu', action='store_true', help='Utilize gpu to train')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available:densenet121, vgg16]')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units for  fc layer')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('data_dir', default='flowers', help='dataset directory')
    parser.add_argument('--save_dir' , type=str, default='./', help='checkpoint directory path')
    args = parser.parse_args()
    return args    
