import torchvision.transforms as T
from matplotlib import pyplot as plt
from numpy import ndarray
from PIL.Image import Image
from torch import float32
from torch import stack
from torch import tensor

from .const import CLASSES
from .const import COLORS


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * tensor([img_w, img_h, img_w, img_h], dtype=float32)
    return b


def check_image_size(img: ndarray):
    assert (
        img.shape[-2] <= 1600 and img.shape[-1] <= 1600
    ), "demo model only supports images up to 1600 pixels on each side"


def plot_results(pil_img: Image, prob, boxes, figsize=(16, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cl = p.argmax()
        text = f"{CLASSES[cl]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def get_transform():
    return T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
