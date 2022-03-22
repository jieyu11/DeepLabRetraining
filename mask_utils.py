import numpy as np
from PIL import Image
import cv2
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def refine_mask(mask, img, segimg=None, foreground_index=255, params=[7, 2, 0, 1]):
    # Use grab cut to refine the mask
    k_size, erode_it, dilate_it, gc_it = params
    kernel = np.ones((k_size, k_size))
    if segimg is not None:
        if segimg.shape != mask.shape:
            logger.warning("Segmentation image shape %s != %s mask image shape" %
                           (str(segimg.shape), str(mask.shape)))
            w, h = mask.shape
            segimg = segimg[0:w, 0:h]

        dilated_mask = 255 * ((mask + segimg) > 0)
        eroded_mask = 255 * (segimg > 0)
        eroded_mask += cv2.dilate(eroded_mask.astype('uint8'),
                                  np.ones((10, 10)), iterations=2) * mask
        eroded_mask = 255 * (eroded_mask > 0)
        gc_it = 3
    else:
        dilated_mask = cv2.dilate(
            mask, kernel, iterations=dilate_it)  # outer boundary
        eroded_mask = cv2.erode(
            mask, kernel, iterations=erode_it)  # inner boundary

    if np.count_nonzero(dilated_mask) == 0 or np.count_nonzero(eroded_mask) == 0:
        logger.warning("Dilated or eroded mask not found! Return raw.")
        return mask

    # these have to be passed to grabCut in order for it to function
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    gc_mask = np.zeros_like(mask) + cv2.GC_BGD
    # use foreground_index (15 for selftrain) since that represents human!
    gc_mask[dilated_mask == foreground_index] = cv2.GC_PR_BGD
    gc_mask[eroded_mask == foreground_index] = cv2.GC_FGD
    new_mask, bgdModel, fgdModel = cv2.grabCut(
        img.astype('uint8'), gc_mask.astype('uint8'), None, bgdModel, fgdModel,
        gc_it, cv2.GC_INIT_WITH_MASK)

    final_mask = ((new_mask == cv2.GC_FGD) + (new_mask == cv2.GC_PR_FGD))*255

    return final_mask


def keypoint_seg_refine(mask, segimg, keypoints):
    """
    For the given mask, go through the keypoints and the pixels in seg-img 
    surrounding the given keypoints to
    keypoints is dict expected with keys:
        nose
        left_eye
        right_eye
        left_ear
        right_ear
        left_shoulder
        right_shoulder
        left_elbow
        right_elbow
        left_wrist
        right_wrist
        left_hip
        right_hip
        left_knee
        right_knee
        left_ankle
        right_ankle
    use the average of left/right_shoulder to left/right_ankle distances as a
    standard distance. Define this values as: d_sh2ank.
    To recover pixels around the ankle, any pixel around left/right_ankle with
    less than 0.5 knee to ankle distance
    """
    def calc_distance(p0, p1):
        # expecting p0 and p1 are arrays of at least 2 numbers
        # indicating the x and y positions
        # p0 = [x0, y0], p1 = [x1, y1]
        assert len(p0) >= 2 and len(p1) >= 2
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        d = np.sqrt(dx*dx + dy*dy)
        return d

    assert mask.shape == segimg.shape, "Mask and seg must have same shape"
    for key in ["left", "right"]:
        assert key+"_ankle" in keypoints and key+"_hip" in keypoints
        logger.debug("KP ankle location: %s" % str(keypoints[key+"_ankle"]))
        logger.debug("KP hip location: %s" % str(keypoints[key+"_hip"]))
        if np.any(np.isnan(keypoints[key+"_ankle"])):
            logger.warning("%s_ankle is not found. Skip." % key)
            continue

        if np.any(np.isnan(keypoints[key+"_hip"])):
            d = int(mask.shape[0] * 0.3)
            logger.warning("%s hip not found, use d:%d" % (key, d))
        else:
            d = calc_distance(keypoints[key+"_ankle"],
                              keypoints[key+"_hip"])
        logger.debug("Distance ankle to hip (%s): %d" % (key, int(d)))
        d = int(d * 0.25)
        ank_x, ank_y = keypoints[key+"_ankle"][0:2]
        xmin, xmax = max(0, ank_x - d), ank_x+d+1
        ymin, ymax = max(0, ank_y - d), ank_y+d+1
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
        # x and y are swapped
        segpart = segimg[ymin:ymax, xmin:xmax]
        # update the mask
        maskpart = mask[ymin:ymax, xmin:xmax]
        mask[ymin:ymax, xmin:xmax] = 255 * (segpart + maskpart > 0)
    return mask


def remove_noise(mask):
    maskC = np.array(mask, dtype=np.uint8)
    cnts, _ = cv2.findContours(
        maskC, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    maskF = np.ones(maskC.shape, dtype=np.uint8)
    if len(cnts) != 0:
        cmax = max(cnts, key=cv2.contourArea)
        cv2.drawContours(maskF, [cmax], -1, 0, -1)
        mask = Image.fromarray((1-maskF)*255)
        # element-wise and to prevent some interior areas from being filled in
        mask = mask * (maskC > 0)
        return mask
    else:
        logger.warning("In remove_noise(mask), contours not found!")
        return mask


def image_size_conversion(label_img, raw_img_shape):
    """
    The output labeled image is of size 160 x 160 pixels defined by the algorithm.
    To convert the output labeled images back to the original image size, they need
    to be cropped first and then resized. For example, for a wide image (width >
    height), the bottom part of the labeled image is meaningless and needs to be cut
    out. Similar treatment is needed for tall images.

    Parameters:
        label_img: (np.array of 3 dimensions size by default (160, 160, 3)) by inferencing
            the algorithm.
        raw_img_shape: (int, int, ?) the raw input image's shape with its height and width.

    Returns:
        cropped_img: (np.array of 3 dimensions (raw_height, raw_width, 3)), which
            is the cropped image of the label_img to match the size of the raw image.
    """
    raw_height, raw_width = raw_img_shape[0], raw_img_shape[1]
    # check if it is a wide image
    wideimg = raw_width > raw_height
    fwh = raw_height/raw_width if wideimg else raw_width/raw_height

    height, width = label_img.shape[0], label_img.shape[1]
    if wideimg:
        # cut the bottom to make the image wide
        height = int(height * fwh)
    else:
        width = int(width * fwh)
    # crop the initial label image to cut off the meaningless parts around
    # bottom or right side
    label_img = label_img[0:height, 0:width]
    label_img = cv2.resize(
        label_img.astype('float32'), (raw_width, raw_height)).astype('uint8')

    logger.debug("Converted label image size: %s " % str(label_img.shape))
    return label_img.astype('uint8')


def image_2Dto3D(label_img):
    """
    Convert 2D image file with shape (w, h) to 3D image (w, h, 3).
    """
    # resize the image to match the input image shape
    if len(label_img.shape) == 2:
        label_img = np.repeat(label_img[:, :, np.newaxis], 3, axis=2)
    return label_img.astype('uint8')


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
