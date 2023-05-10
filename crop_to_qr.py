import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pyzbar.pyzbar import decode


def crop_image(path, fig=None, fig2=None):
    img = cv2.imread(path)

    # aspect = image.shape[0] / image.shape[1]
    aspect = 1  # image.shape[0]/image.shape[1]
    barcodes = decode(img)
    # print(barcodes)

    if fig is not None:
        fig.clf()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fig.show()

    if barcodes is not None and len(barcodes) == 1:
        rect_p = barcodes[0].rect
        poly_p = barcodes[0].polygon
        sides = np.roll(poly_p, 0) - np.roll(poly_p, 2)

        # scewed poly
        poly_2 = np.array(poly_p)
        poly_2[0] += sides[0] - sides[1]
        poly_2[1] += sides[0] + (.5 * sides[1]).astype(int)
        poly_2[3] += 2 * sides[2] + sides[3]
        poly_2[2] += 2 * sides[2] - (.5 * sides[3]).astype(int)

        # rect around scewed
        rect_px4_c = np.mean(poly_p, axis=0)
        rect_px4_0 = rect_px4_c - np.mean([rect_p.width, rect_p.height]) * np.array([2, 2 * aspect])
        rect_px4_d = 3 * np.array([rect_p.width, rect_p.height * aspect])

        # 2x base bounding box
        #    rect_2_0 = np.min(poly_2[:,0]), np.min(poly_2[:,1])
        #   rect_2_wh = np.array([np.max(poly_2[:, 0]), np.max(poly_2[:, 1])])-np.array(rect_2_0)
        scale = 3
        rect_2_wh = int(scale * np.mean([rect_p.width, rect_p.height]))
        rect_2_0 = (np.mean(poly_p, axis=0) - np.array(2 * [rect_2_wh]) / 2).astype(int)

        #    poly_2[1] += sides[0]+sides[1]
        # img = cv2.imread("test.png")

        """
        plt.gca().add_patch(
             patches.Polygon(np.array(poly_p), linewidth=1, edgecolor='b', facecolor='none'))
    
        plt.gca().add_patch(
            patches.Polygon(np.array(poly_2), linewidth=1, edgecolor='m', facecolor='none'))
    
        plt.gca().add_patch(
            patches.Rectangle(rect_px4_0, *rect_px4_d, linewidth=1, edgecolor='r',
                              facecolor='none'))
        """
        if fig is not None:
            fig.gca().add_patch(
                patches.Rectangle((rect_p.left, rect_p.top), rect_p.width, rect_p.height, linewidth=1, edgecolor='r',
                                  facecolor='none'))
            fig.gca().add_patch(
                patches.Rectangle(rect_2_0, rect_2_wh, rect_2_wh, linewidth=3, edgecolor='m',
                                  facecolor='none'))

        w, h = img.shape[:2]
        a, b = int(rect_2_0[1]), int(rect_2_0[1] + rect_2_wh)
        if a < 0:
            b = b - a
            a = 0
        if b >= w:
            overshoot = b - w + 1
            a = a - overshoot
            b = w - 1

        c, d = int(rect_2_0[0]), int(rect_2_0[0] + rect_2_wh)
        if c < 0:
            d = d - c
            c = 0
        if d >= h:
            overshoot = d - h + 1
            a = a - overshoot
            d = h - 1

        # img2 = img[int(rect_2_0[1]):int(rect_2_0[1] + rect_2_wh),
        #       int(rect_2_0[0]):int(rect_2_0[0] + rect_2_wh), :]

    else:
        sq = np.min(img.shape[:2])
        sq0 = np.abs(np.diff(img.shape[:2]) / 2).astype(int)[0]
        if fig is not None:
            fig.gca().add_patch(
                patches.Rectangle((0, sq0), sq, sq, linewidth=3, edgecolor='m',
                                  facecolor='none'))
        a, b = sq0, (sq0 + sq)
        c, d = 0, -1
        # img2 = img[sq0:(sq0 + sq), :, :]

    if fig2 is not None:
        fig2.clf()
        plt.imshow(cv2.cvtColor(img[a:b, c:d], cv2.COLOR_BGR2RGB))
    return a, b, c, d


if __name__ == "__main__":
    from qr_torch.qrdata import QrData, SubSet, Label

    data = QrData(SubSet.TRAIN)

    ind = random.randint(0, len(data))
    blob_name, label = data.blobs.iloc[ind]
    cache_path = str(data.img_dir.joinpath(f"{Label(label).prefix()}", blob_name.split("/")[1]))
    cache_path = r"C:\Users\Matjaz\.qr_dataset\qr-device\FLEX0001QN_2021-03-21T22-31-19.jpg"
    bbox = crop_image(cache_path, plt.figure(1), None)
    print(bbox)
