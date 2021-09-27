from Modules import  *

def Imageprocessing(image):
    
    im = Image.open(image)

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        # Preprocess the image
    img_tensor = preprocess(im)
    return img_tensor.numpy()





def ImagePrediction(image_path, model, topks, device,indexClass):
        img=Imageprocessing(image_path)
        img = Image.open(image_path)
        img=torch.FloatTensor([img])
        model.eval()
        output=model(img.to(device))
        probability=torch.exp(output.cpu())
        top_p,top_c = probability.topk(topks,dim=1)
#         print(type(idx_to_class))
        top_class = [indexClass.get(x) for x in top_c.numpy()[0]]
        return top_p,top_class