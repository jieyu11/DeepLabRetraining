import os
import cv2
import json
import argparse
import logging
import numpy as np
from inference_rawmodel import DeepLabV3

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_frames(video_filename, nframes=6, nskip=15):
    frames_dict = {}
    # Opens the Video file
    cap = cv2.VideoCapture(video_filename)
    idx, iframe = 0, 0
    while(cap.isOpened() and iframe < nframes):
        ret, frame = cap.read()
        if not ret:
            break
        if len(frames_dict) < nframes and idx % nskip == 0:
            iframe += 1
            frames_dict[idx] = frame
        # cv2.imwrite('kang'+str(i)+'.jpg',frame)
        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Selected %d frames from %s" %
                (len(frames_dict), video_filename))
    return frames_dict


def get_masks(frames_dict):
    """
    Given the frames, make the bgsegmentation inference and get the masks with True for
    human and False for backgrounds.
    """
    masks_list = []
    model = DeepLabV3("deeplabv3_resnet50")
    for frame in frames_dict.values():
        masks_list.append(model.segment_image(frame))
    return masks_list


def get_boundary(masks_list, enlarge=0.1):
    """
    Given the masks in the masks list, return the bounding box, xmin,ymin to xmax,ymax.
    Parameters:
        masks_list: (np.array list), masks are represented with np.array
        enlarge: (float) in [0., 1.], adding edge to the bounding box by xmin -= 0.1 * xlen
            because the persons are standing and in general xlen << ylen, a factor of two
            is applied to y (width) ymin -= 0.1 * ylen * 2

    Returns:
        tuple of (xmin, ymin, xmax, ymax), where x is height, y is width
    """
    assert 0 <= enlarge < 1.0, "parameter enlarge: %.2f must in [0, 1)" % enlarge
    assert len(masks_list) and len(
        masks_list[0].shape) >= 2, "Masks must not be empty"
    npix_x, npix_y = masks_list[0].shape[0:2]
    logger.info("Frames initial shape: (%d, %d)" % (npix_x, npix_y))
    bounds = [99999, 0, 99999, 0]
    for mask in masks_list:
        # indices with non-zero values in mask
        x_idx, y_idx = np.nonzero(mask)
        bounds = [
            min(bounds[0], np.percentile(x_idx, 1)),
            max(bounds[1], np.percentile(x_idx, 99)),
            min(bounds[2], np.percentile(y_idx, 1)),
            max(bounds[3], np.percentile(y_idx, 99))
        ]
    xlen, ylen = bounds[1] - bounds[0], bounds[3] - bounds[2]
    assert xlen > 0 and ylen > 0, "length in x and y must be positive"
    bounds[0] = max(int(bounds[0] - xlen * enlarge), 0)
    bounds[1] = min(int(bounds[1] + xlen * enlarge), npix_x)
    bounds[2] = max(int(bounds[2] - ylen * enlarge * 2), 0)
    bounds[3] = min(int(bounds[3] + ylen * enlarge * 2), npix_y)
    logger.info("Frames boundary X: (%d, %d), Y: (%d, %d)" %
                (bounds[0], bounds[1], bounds[2], bounds[3]))
    xlen, ylen = bounds[1] - bounds[0], bounds[3] - bounds[2]
    logger.info("Cropped box X (height): %d, Y (width): %d" % (xlen, ylen))
    return bounds


def image_overlay(image, xmin, xmax, ymin, ymax):
    alpha = .5
    img_ = cv2.UMat(np.array(image, dtype=np.uint8))
    img_bound = image
    img_bound[xmin:xmax, ymin:ymax, :] = 255
    img_mix = cv2.addWeighted(img_bound, alpha, img_, 1 - alpha, 0)
    return img_mix


def main(args):
    outdir = os.path.dirname(args.output)
    os.makedirs(outdir, exist_ok=True)
    assert os.path.exists(args.input), "Input %s is not found!" % args.input
    frames_dict = get_frames(args.input, args.nframes, args.nskip)
    masks_list = get_masks(frames_dict)
    bounds = get_boundary(masks_list, args.bound_enlarge)
    out_dict = {"xmin": bounds[0], "xmax": bounds[1],
                "ymin": bounds[2], "ymax": bounds[3]}
    with open(args.output, 'w') as f:
        json.dump(out_dict, f)
    logger.info("Boundary output saved to: %s" % args.output)
    # save the images
    if args.viz:
        vizdir = outdir if args.vizdir is None else args.vizdir
        vizdir = os.path.join(vizdir, "bound")
        os.makedirs(vizdir, exist_ok=True)
        xmin, xmax = out_dict["xmin"], out_dict["xmax"]
        ymin, ymax = out_dict["ymin"], out_dict["ymax"]
        for idx, frame in frames_dict.items():
            imgname = os.path.join(vizdir, "frame%06d.jpg" % idx)
            frame = image_overlay(frame, xmin, xmax, ymin, ymax)
            cv2.imwrite(imgname, frame)
            logger.info("Saving image: %s" % imgname)


if __name__ == '__main__':
    """
    To run, it needs input indexed images folder and output folder:
        python3 video_crop.py -i /source/video.mp4 \
                -o /bound/output.json \
                [--nframes 6 --nskip 15 --bound-enlarge 0.1 --viz]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", default=None, type=str,
        required=True, help="Your input video file name.",
    )
    parser.add_argument(
        "--output", "-o", default=None, type=str, required=True,
        help="Your output text file name (e.g. out.json).",
    )
    parser.add_argument(
        "--nframes", "-n", default=6, type=int,
        required=False, help="Number of frames to be used for segmentation",
    )
    parser.add_argument(
        "--nskip", "-s", default=15, type=int,
        required=False, help="Number of frames to be skipped every time.",
    )
    parser.add_argument(
        "--bound-enlarge", "-b", default=0.10, type=float,
        required=False, help="Percentage of the boundary to be enlarged.",
    )
    parser.add_argument(
        "--viz", action="store_true", required=False,
        help="Use --viz to save vizulization images for bounds.",
    )
    parser.add_argument(
        "--vizdir", type=str, default=None, required=False,
        help="Set vizdir for visualization images with bounding box.",
    )
    args = parser.parse_args()
    main(args)
