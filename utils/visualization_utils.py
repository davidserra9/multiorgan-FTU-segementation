import cv2
import os
import numpy as np
from PIL import Image
from src.tools.utils import read_mask

def read_img_and_mask(img_name, path_db=os.path.join('../../data', 'hubmap-organ-segmentation'), show=False):
    """
    Read the image and the mask from the dataset.
    :param img_name: name of the image to read and to extract the polygons from
    :param show: if True, show the image with the FTUs encircled
    :return: image in rgb, binary mask of the FTUs
    """

    # get the paths were imgs and masks are located
    path_img = os.path.join(path_db, 'train_images', f'{img_name}.tiff')
    path_mask = os.path.join(path_db, 'train_annotations', f'{img_name}.json')

    # control if path exists
    if not os.path.exists(path_img):
        raise Exception('Path to imgs does not exist: {}'.format(path_img))

    # read the image
    img = cv2.imread(path_img)

    # get the associated mask
    mask = read_mask(path_mask)

    # in here will be stored all the shapes of the
    polygons = []
    # get connected components from mask. mask is a list of lists where each list is a connected component
    for component in mask:
        # get the polygon of the component using convex hull
        polygon = cv2.convexHull(np.array(component))
        if show:
            # draw the polygon
            cv2.polylines(img, [polygon], True, (0, 0, 255), thickness=3)
        # append the polygon to the list
        polygons.append(polygon)

    # create a mask from the polygons
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # for each polygon in the list
    for polygon in polygons:
        # all the points of the polygon and inside the points of the polygon, will be a 1 of the mask
        cv2.fillPoly(mask, [polygon], 1)

    # visualize image
    if show:
        # reduce img to show it in a smaller window
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # show the image
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img, mask


def visualize_output_unet_medical(output):
    """
    Visualize the output of the UNET model.
    :param output: output of the UNET model
    :return: image with the output of the UNET model
    """
    output = output[0].cpu().numpy()
    output = output.transpose(1, 2, 0)
    output = output.squeeze()
    output = (output - output.min()) / (output.max() - output.min())
    output = output * 255
    output = output.astype(np.uint8)
    output = Image.fromarray(output)
    output.show()


if __name__ == "__main__":
    img, mask = read_img_and_mask(img_name='203', show=True)
    print('finished')