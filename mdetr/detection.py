import argparse
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tt
from PIL import Image
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from transformers import CharSpan
import portion
from portion import Interval

from hubconf import _make_detr

torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = tt.Compose([
    tt.Resize(800),
    tt.ToTensor(),
    tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size) -> torch.Tensor:
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]
]


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] *
            (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c]
        )
    return image


def plot_results(pil_img: Image, scores, boxes: List[List[int]], labels: List[str], masks=None):
    plt.figure(figsize=(16, 10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
        masks = [None] * len(scores)
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for score, (xmin, ymin, xmax, ymax), label, mask, color in zip(scores, boxes, labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        ax.text(xmin, ymin, f'{label}: {score:0.2f}', fontsize=15, bbox=dict(facecolor=color, alpha=0.8),
                fontname='Hiragino Maru Gothic Pro')

        if mask is None:
            continue
        np_image = apply_mask(np_image, mask, color)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor='none', edgecolor=color)
            ax.add_patch(p)

    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig('output.png')
    plt.show()


# def add_res(results, ax):
#     # for tt in results.values():
#     if True:
#         bboxes = results['boxes']
#         labels = results['labels']
#         scores = results['scores']
#         # keep = scores >= 0.0
#         # bboxes = bboxes[keep].tolist()
#         # labels = labels[keep].tolist()
#         # scores = scores[keep].tolist()
#     # print(torchvision.ops.box_iou(tt['boxes'].cpu().detach(), torch.as_tensor([[xmin, ymin, xmax, ymax]])))
#
#     colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']
#
#     for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
#         ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
#         cls_name = ll if isinstance(ll, str) else CLASSES[ll]
#         text = f'{cls_name}: {ss:.2f}'
#         print(text)
#         ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))


def plot_inference(im: Image, caption: str, model: nn.Module):
    # mean-std normalize the input image (batch-size: 1)
    if torch.cuda.is_available():
        img: torch.Tensor = transform(im).unsqueeze(0).cuda()  # (1, ch, H, W)
    else:
        img: torch.Tensor = transform(im).unsqueeze(0)

    # propagate through the model
    memory_cache = model(img, [caption], encode_and_save=True)
    # dict keys: 'pred_logits', 'pred_boxes', 'proj_queries', 'proj_tokens', 'tokenized'
    # pred_logits: (1, cand, seq)
    # pred_boxes: (1, cand, 4)
    # proj_queries: (1, cand, 64)
    # proj_tokens: (1, 28, 64)
    # tokenized: BatchEncoding
    outputs: dict = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

    # keep only predictions with 0.7+ confidence
    probs = 1 - outputs['pred_logits'].softmax(dim=2)[0, :, -1].cpu()  # (cand)
    keep = (probs > 0.8)  # (cand)

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)  # (kept, 4)

    # Extract the text spans predicted by each box
    # (140, 2)
    positive_tokens = torch.nonzero(outputs['pred_logits'].cpu()[0, keep].softmax(-1) > 0.1).tolist()
    predicted_spans: Dict[int, Interval] = defaultdict(lambda: portion.empty())
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            try:
                span: CharSpan = memory_cache['tokenized'].token_to_chars(0, pos)
            except TypeError:
                continue
            predicted_spans[item] |= portion.closedopen(span.start, span.end)

    labels: List[str] = [
        ','.join(''.join(caption[j] for j in portion.iterate(i, step=1)) for i in predicted_spans[k])
        for k in sorted(predicted_spans.keys())
    ]
    plot_results(im, probs[keep], bboxes_scaled.tolist(), labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, help='Path to trained model.')
    # parser.add_argument('--image-dir', '--img', type=str, help='Path to the directory containing images.')
    parser.add_argument('--image-path', '--img', type=str, help='Path to the images file.')
    parser.add_argument('--text', type=str, default='5 people each holding an umbrella',
                        help='split text to perform grounding.')
    # parser.add_argument('--dialog-ids', '--id', type=str, help='Path to the file containing dialog ids.')
    args = parser.parse_args()

    # model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True,
    #                                       return_postprocessor=True)
    model = _make_detr(backbone_name='timm_tf_efficientnet_b3_ns', text_encoder='xlm-roberta-base')
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # url = "http://images.cocodataset.org/val2017/000000281759.jpg"
    # web_image = requests.get(url, stream=True).raw
    # im = Image.open(web_image)
    image = Image.open(args.image_path)
    plot_inference(image, args.text, model)


if __name__ == '__main__':
    main()