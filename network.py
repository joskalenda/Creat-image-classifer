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
