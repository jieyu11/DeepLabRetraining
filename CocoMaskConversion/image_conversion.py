import os
import glob
import argparse
import numpy as np
import cv2
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_image(img, h, w, border=cv2.BORDER_CONSTANT, mode="pad"):
    """
    For a given image: img, pad it to be size (w, h)
    border: cv2.BORDER_CONSTANT filled with [0,0,0] or
        cv2.BORDER_REPLICATE, more details find:
        https://docs.opencv.org/4.5.3/d3/df2/tutorial_py_basic_ops.html
    mode: pad or resize

    Returns:
        resized image with (h, w)
    """
    assert mode in ["pad", "resize"]
    # image resize
    if mode == "resize":
        return cv2.resize(img, (w, h))

    # image padding
    h0, w0, _ = img.shape
    if h0 > h or w0 > w:
        logger.debug("original image shape: %s" % str(img.shape))
        scale = min(h / h0, w / w0)
        h1, w1 = int(h0 * scale), int(w0 * scale)
        img = cv2.resize(img, (w1, h1))
        logger.debug("scaled image shape: %s" % str(img.shape))
    h0, w0, _ = img.shape
    assert h0 <= h and w0 <= w, "image size requirement"
    dw, dh = w - w0, h - h0
    img = cv2.copyMakeBorder(img, top=dh//2, bottom=dh-dh//2, left=dw//2,
                             right=dw-dw//2, borderType=border,
                             value=[0, 0, 0])
    return img


def convert_image_color(img, rgb_original, rgb_converted, inverse=False):
    """
    Function to convert original color pixels into other colors.

    Parameters:
        img: is the image (numpy.array) with shape: (w, h, 3)
        rgb_original_list: a list of 1D array with 3 values, e.g.
            [[128, 0, 128], [0, 64, 128]] pixels with these colors will be
            replaced.
        rgb_converted: 1D array with 3 values, e.g. [255, 255, 255], the pixels
            above are replaced with this color.
        inverse: bool, if True, then pixels WITHOUT rgb_original is
            replaced with rgb_converted
    """
    assert len(rgb_converted) == 3, "rgb length of 3"
    assert len(rgb_original) == 3, "rgb length of 3"
    data = np.array(img)
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    # Original value
    r1, g1, b1 = rgb_original[0], rgb_original[1], rgb_original[2]
    if inverse:
        mask = (red != r1) | (green != g1) | (blue != b1)
    else:
        mask = (red == r1) & (green == g1) & (blue == b1)
    # Value that we want to replace it with
    r2, g2, b2 = rgb_converted[0], rgb_converted[1], rgb_converted[2]
    data[:, :, :3][mask] = [r2, g2, b2]
    return data


def main(args):
    imgtype = args.type.lower()
    assert imgtype in ["mask", "image"], "image type is mask or image"
    os.makedirs(args.outputfolder, exist_ok=True)
    imgnames = glob.glob(args.inputfolder + "/*.png")
    for idx, name in enumerate(imgnames):
        basename = os.path.basename(name)
        img = cv2.imread(name)
        if imgtype == "mask":
            # human: [36, 231, 253] to [255, 255, 255]
            # background: [84  1 68] to [0, 0, 0]
            img = convert_image_color(img, [84, 1, 68], [0, 0, 0])
            img = convert_image_color(img, [36, 231, 253], [255, 255, 255])
        img = resize_image(img, args.height, args.width, mode=args.scale)
        outname = os.path.join(args.outputfolder, basename)
        cv2.imwrite(outname, img)
        if idx % 100 == 0:
            logger.info("%04d, Initial image: %s" % (idx, name))
            logger.info("%04d, Converted image: %s" % (idx, outname))
    logger.info("Total images: %d" % (idx+1))


if __name__ == '__main__':
    """
    To run, it needs input indexed images folder and output folder:
        python3 image_conversion.py -i /source/images/folder \
                -o /resized/images/folder -t mask [OR: image] \
                [opt: --width 480 --height 640]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfolder", "-i", default=None, type=str, required=True,
        help="Your inputfolder containing all Pinocchio run outputs.",
    )
    parser.add_argument(
        "--outputfolder", "-o", default=None, type=str, required=True,
        help="Your output folder.",
    )
    parser.add_argument(
        "--type", "-t", default=None, type=str, required=True,
        help="Your image type: mask OR image",
    )
    parser.add_argument(
        "--height", default=640, type=int, required=False,
        help="Your target image height, default=640.",
    )
    parser.add_argument(
        "--width", default=480, type=int, required=False,
        help="Your target image width, default=480.",
    )
    parser.add_argument(
        "--scale", default="resize", type=str, required=False,
        help="Your scale mode to match the required size: resize or pad",
    )
    args = parser.parse_args()

    main(args)
