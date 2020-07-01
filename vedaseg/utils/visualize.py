import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_image(img_dir, order='BGR'):
    sample = cv2.imread(img_dir)
    assert sample is not None, f'sample from {img_dir} is empty'
    if order.upper() == 'RGB':
        return cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    return sample


def get_plot(img_, pd, vis_mask=False, vis_contour=True, output_dir=None,
             inverse_color_channel=False, n_class=2, color_name='ocean'):
    if inverse_color_channel:
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    else:
        img = img_.copy()

    cmap = plt.cm.get_cmap(color_name, n_class)
    plt.figure(figsize=(16, 12))
    plt.suptitle('X-ray image inspection: Knife')
    plt.tight_layout()

    plt.subplot(121)
    plt.title('Input Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(122)
    plt.title('Image with Prediction')
    img = process_res(img, pd, cmap, class_id=None, vis_mask=vis_mask,
                      vis_contour=vis_contour)
    plt.imshow(img)
    plt.axis('off')
    if output_dir is None:
        plt.show()
        return plt
    else:
        plt.savefig(output_dir, transparent=True)
        plt.cla()
        plt.clf()
        plt.close('all')


def process_res(img, pd, cmap, class_id=None, vis_mask=False, vis_contour=True):
    if class_id is None:
        class_id = np.sort(np.unique(pd))
        if class_id[0] == 0:
            class_id = class_id[1:]
    elif not isinstance(class_id, list):
        class_id = [class_id]

    for idx, class_ in enumerate(class_id):
        color = np.array(cmap(idx)[:3]) * 255
        if vis_mask:
            img = image_overlay(img, pd, color, class_id=class_)
        if vis_contour:
            img = get_contours(img, pd, color, class_id=class_)
    return img


def image_overlay(image, mask, color, class_id=1):
    overlay_img = image.copy()
    heat_map = np.zeros_like(image)
    heat_map[mask == class_id] = color
    fin = cv2.addWeighted(heat_map, 0.8, overlay_img, 0.8, 0)
    return fin


def get_contours(image, mask, color, class_id=1):
    contour_img = image.copy()
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    contour_mask[mask == class_id] = 1
    res = cv2.findContours(contour_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[1] if len(res) == 3 else res[0]
    cv2.drawContours(contour_img, contours, -1, color, 3)
    return contour_img
