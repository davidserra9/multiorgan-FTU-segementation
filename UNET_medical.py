import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tools.visualize import visualize_output_unet_medical

# ojo este paper: https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf

# https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
def load_unet_medical(init_features=32, pretrained=True):
    """
    Loads a pretrained UNet model for medical image segmentation.
    :param init_features: Number of features in the first convolution layer.
    :param pretrained: If True, loads a pretrained model.
    :return: model: A PyTorch model.
    """
    net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                         init_features=init_features, pretrained=pretrained)

    return net


def inference(net, img):
    """
    Performs inference on a single image. The image is converted to a PIL image, then resized to the size of the model,
    and then converted to a tensor. The model is then run on the tensor. The output is then converted to a numpy array.
    :param net: A PyTorch model.
    :param img: A numpy array or PIL image.
    """
    m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(2048, 2048)),
        transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        net = net.to('cuda')

    with torch.no_grad():
        output = net(input_batch)

    visualize_output_unet_medical(output)


if __name__ == "__main__":
    model = load_unet_medical()
    data = Image.open('/home/francesc/PycharmProjects/kaggle/HuBMAP+HPA-Hacking-the-Human-Body/data/hubmap-organ-segmentation/train_images/203.tiff')
    inference(model, data)