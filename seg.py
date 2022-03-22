import os
import numpy as np
import argparse
import h5py
import cv2
import logging
from tqdm import tqdm
from inference import DeepLabV3 as DeepLabV3Retrained
from inference_rawmodel import DeepLabV3 as DeepLabV3Raw

from libs_internal.io.h5 import (load_h5_group_names, load_h5_group_vals)
from mask_utils import (refine_mask, remove_noise)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLabModel(object):
    """
      Class to load deeplab model (retrained or original) and run inference.
    """

    def __init__(self, config):
        """Creates and loads pretrained deeplab model."""
        self.input_h5 = config.get("input_h5", "PreProcessing.h5")
        # use frames in keyp_h5 for segmentation
        self.keyp_h5 = config.get("keyp_h5", None)
        # if keyp_h5 is not given, then use the frames from input h5
        self.keyp_h5 = self.input_h5 if not self.keyp_h5 else self.keyp_h5
        self.output_h5 = config.get("output_h5", "BgSegmentation.h5")
        # for retraining, using the model filename
        self.modelfile = config.get(
            "modelfile", "weights_deeplabv3_retrain_coco_width480_height640.pt")
        # for original model, use "deeplabv3_resnet50", or "deeplabv3_resnet101",
        # or "deeplabv3_mobilenet_v3_large"
        # whether or not making visulization into output
        self.viz = config.get("viz", True)
        # whether or not do a refinement after segmentation is done
        self.refinement = config.get("refinement", False)
        retrained = config.get("retrained", True)
        if retrained:
            # width and height are used for model retraining
            width = config.get("width", 480)
            height = config.get("height", 640)
            self.model = DeepLabV3Retrained(self.modelfile, width, height)
        else:
            self.model = DeepLabV3Raw(self.modelfile)

    def _load_h5_files(self):
        assert os.path.exists(self.keyp_h5), "%s not found" % self.keyp_h5
        frames = load_h5_group_names(self.keyp_h5)
        assert len(frames) > 0, "No frames are found! Check %s!" % self.keyp_h5

        assert os.path.exists(self.input_h5), "%s not found" % self.input_h5
        imglist = load_h5_group_vals(self.input_h5, 'data', frames)
        # image shape is (height, width, 3), needed for saving video!!
        self.img_height, self.img_width, _ = imglist[0].shape
        self.frame_dict = {name: img for name, img in zip(frames, imglist)}

    def run(self):
        """
        Run the background vs person segmentation
        """
        # load the input files into memory
        self._load_h5_files()
        mask_dict = {}
        # looping over the frames and do background segmentation inferences
        for name, frame in self.frame_dict.items():
            # masks are True or False np.array
            mask = self.model.segment_image(frame)
            if self.refinement:
                # refine mask with refinement and noise removal
                mask = self.refine(mask, frame)
            # update mask with bool
            mask_dict[name] = mask

        self.save(mask_dict)

    def save(self, mask_dict):
        """
        given the output masks after running the segmentation, 
        Save the masks to output file!
        """
        with h5py.File(self.output_h5, 'w') as f:
            group = f.create_group("info")
            dset = group.create_dataset(name=np.string_('BgSegmentation'),
                                        data=[np.string_('boolean')],
                                        compression="gzip",
                                        compression_opts=9)
            group = f.create_group("data")

            if self.viz:
                path, filename = os.path.split(self.output_h5)
                basename = filename.split(".")[0]
                folder = os.path.join(path, basename)
                os.makedirs(folder, exist_ok=True)
                filename = os.path.join(folder, basename+".mp4")
                video = cv2.VideoWriter(
                    filename, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                    (self.img_width, self.img_height)
                    )
            for name, mask in tqdm(mask_dict.items()):
                if self.viz:
                    alpha = .5
                    img_ = cv2.UMat(
                        np.array(self.frame_dict[name], dtype=np.uint8))
                    # height, width = self.frame_dict[name].shape[0:2]
                    mask_ = np.zeros((self.img_height, self.img_width, 3),
                                     dtype=np.uint8)
                    mask_white = (mask > 0).astype("uint8") * 255
                    mask_[:, :, 0] = mask_white
                    mask_[:, :, 1] = mask_white
                    mask_[:, :, 2] = mask_white
                    dst_ = cv2.addWeighted(mask_, alpha, img_, 1 - alpha, 0)
                    video.write(dst_)
                dset = group.create_dataset(
                    name=(name),
                    data=np.array(mask, dtype=bool),
                    shape=(self.img_height, self.img_width),
                    maxshape=(self.img_height, self.img_width),
                    compression="gzip",
                    compression_opts=9,
                    dtype='float32')
            if self.viz:
                video.release()

    def refine(self, mask, img):
        """
        refine the mask using grabcut and noise removal given the initial
        image.
        Parameters:
            mask: np.array, initial mask
            img: np.array, initial image
        Returns:
            mask: np.array, after refinement
        """
        # convert mask to 0's and 255's
        mask = (mask > 0).astype("uint8") * 255
        mask = refine_mask(np.array(mask, dtype=np.uint8), img)
        # removing noise only whan refine grabcut is already done
        # due to a mask class limitation
        mask = remove_noise(mask)
        # convert mask to bool
        mask = mask > 0
        return mask


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-hdf5', '-i', type=str,
        help="Input HDF5 file that contains images")
    parser.add_argument(
        '--keyp-hdf5', '-k', type=str,
        help="2D Key Points HDF5 file with key points")
    parser.add_argument(
        '--output-hdf5', '-o', type=str,
        help="Output HDF5 file contain masks")
    parser.add_argument(
        '--modelfile', '-m', type=str,
        help="Output HDF5 file contain masks")
    parser.add_argument(
        '--viz', action='store_true',
        help="Save visualization video")
    parser.add_argument(
        '--refinement', action='store_true',
        help="If true, do a refinement after segmentation is done.")
    parser.add_argument(
        '--retrained', action='store_true', 
        help='true for retrained DeepLabV3 model, false raw model')
    args = parser.parse_args()
    return {
        "input_h5": args.input_hdf5,
        "keyp_h5": args.keyp_hdf5,
        "output_h5": args.output_hdf5,
        "viz": args.viz,
        "refinement": args.refinement,
        "retrained": args.retrained,
        "modelfile": args.modelfile,
    }


def main():
    config = get_arguments()
    logger.info("Running with Config: %s" % str(config))
    dv = DeepLabModel(config)
    dv.run()


if __name__ == '__main__':
    main()
