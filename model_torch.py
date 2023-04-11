from torchvision.transforms.functional import to_pil_image

import torch


def model_pre(image, model, preprocess):
    
    img = to_pil_image(image)
    batch = preprocess(img).to('cuda')

    with torch.no_grad():
        info = model([batch])
    
    return info[0]['boxes'].to(torch.int), info[0]['labels'], info[0]['scores']