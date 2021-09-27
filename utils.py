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
