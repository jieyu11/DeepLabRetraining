#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import h5py
import cv2
import logging
import numpy as np
from tqdm import tqdm
from libs_internal.io.h5 import load_h5_group_names
from mask_utils import convert_image_color
logger = logging.getLogger('luigi-interface')
logger.setLevel(logging.INFO)


class H5toImages(object):
    def __init__(self, inputFilePath, outputFolderPath=None):
        """
        Initialization of converting frames from one h5 file to images.
        Parameters:
            - inputFilePath: (str) input .h5 file full path and name
            - outputFolderPath: (str) output folder full path, if it is
                None, then no output is written to files.
        """

        self.inputFilePath = inputFilePath
        self.outputFolderPath = outputFolderPath
        if outputFolderPath is not None:
            os.makedirs(self.outputFolderPath, exist_ok=True)

    def convert(self):
        """
            Read input h5 file and return the image dict. If 
            self.outputFolderPath is not None, then save the images to it. 
            Returns:
                a dict with image name as key and image np.array as value
        """
        imglist = load_h5_group_names(self.inputFilePath)
        with h5py.File(self.inputFilePath, 'r') as infile:
            grp = infile['data']
            # write the image file to a folder
            if self.outputFolderPath is not None:
                for i in tqdm(range(len(imglist))):
                    im = grp[imglist[i]][:]
                    if self.outputFolderPath is not None:
                        cv2.imwrite("%s/%s.png" % (self.outputFolderPath,
                                                   imglist[i]), im)
                logger.info("Out images folder: %s" % self.outputFolderPath)
            imgdict = {name: grp[imglist[i]][:]
                       for i, name in enumerate(imglist)}
            logger.info("Number of input images: %d" % len(imgdict))
            return imgdict


class ImagesToH5(object):
    def __init__(self, viz=True):
        """
        Initialization of converting images to frames in a h5 file.
        Parameters:
            - viz: (bool) whether or not to save a visualization video.
                default to True.
        """

        self.viz = viz

    def convert(self, image_dict, outputFilePath):
        """
            Read images from image_dict (name: np.array) and save them
            to an h5 file together with the meta data
            information including the number of classes, mask, etc.
        """
        # get the width and height of the first image, assuming all images
        # having the same width and height
        for name in image_dict:
            imgmask = image_dict[name][0]
            height, width = imgmask.shape[0], imgmask.shape[1]
            logger.info("Image width=%d, height=%d (assuming for all images)" %
                        (width, height))
            break

        with h5py.File(outputFilePath, 'w') as f:
            group = f.create_group("info")
            dset = group.create_dataset(name=np.string_('BgSegmentation'),
                                        data=[np.string_('boolean')],
                                        compression="gzip",
                                        compression_opts=9)
            group = f.create_group("data")

            if self.viz:
                # Possible path = demo_trigger filename=BgSegmentation_tron.h5
                path, filename = os.path.split(outputFilePath)
                filename = filename.split(".")[0]
                # make the output viz video path to: demo_trigger/BgSegmentation_tron
                path = os.path.join(path, filename)
                os.makedirs(path, exist_ok=True)
                video = cv2.VideoWriter(os.path.join(path, filename+".mp4"),
                                        cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                        (width, height))
            for name in sorted(list(image_dict.keys())):
                mask = image_dict[name][0]
                if self.viz:
                    # convert [1, 1, 1] (person) to color for visualization (e.g. 255, 255, 255: white)
                    # leaving background [0, 0, 0] as is.
                    mask_color = convert_image_color(
                        mask, [1, 1, 1], [255, 255, 255])
                    alpha = .5
                    raw_img = image_dict[name][1]
                    # use BGR2RGB raw_img[:] = raw_img[:, :, ::-1] if reading from image.png
                    dst_ = cv2.addWeighted(
                        mask_color, alpha, raw_img, 1 - alpha, 0)
                    video.write(dst_)

                dset = group.create_dataset(
                    name=name,
                    data=np.array(mask[:, :, 0], dtype=bool),
                    shape=(height, width),
                    maxshape=(height, width),
                    compression="gzip",
                    compression_opts=9,
                    dtype='float32')
            if self.viz:
                video.release()
