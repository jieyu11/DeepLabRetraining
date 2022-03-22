import os
import cv2
import glob
import torch
import argparse
import logging
from torchvision import transforms
from PIL import Image

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class DeepLabV3:
    def __init__(self, model_name):
        # model weights.pt
        assert model_name in ["deeplabv3_resnet50",
                              "deeplabv3_resnet101",
                              "deeplabv3_mobilenet_v3_large"]
        self.model = torch.hub.load('pytorch/vision:v0.7.0',
                                    model_name, pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def segment(self, image_path):
        """
        Make 1 image segmentation inference with the given model, while
        passing the image file full path.

        Arguments:
            image_path: (str), full path to the image file.
        
        return mask (2D np.array)
        """
        assert os.path.exists(image_path), "%s not found!" % image_path
        # Image.open() returns images with 4 channels, convert to 3!
        # working below
        # img = Image.open(image_path).convert('RGB')
        img = cv2.imread(image_path)
        return self.segment_image(img)

    def segment_image(self, img):
        """
        Make 1 image segmentation inference with the given model, while
        passing the image np.array.

        Arguments:
            img: np.array, image content
        
        return mask (2D np.array)
        """
        # logger.info(" - Image shape: %s" % str(img.shape))
        input_tensor = preprocess(img)
        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
            output_predictions = output.argmax(0)
            mask = output_predictions.cpu().detach().numpy()
            # 15 is person, the rest is background and other classes
            # the output image has 21 classes labels, [21, H, W] shape
            return (mask == 15)


def main(args):
    os.makedirs(args.outputfolder, exist_ok=True)
    dv = DeepLabV3(args.modelname)
    imgnames = []
    for ext in ["png", "jpg", "jpeg"]:
        imgnames.extend(glob.glob(args.inputfolder + "/*.%s" % ext))
    for idx, fullname in enumerate(imgnames):
        basename = os.path.basename(fullname)
        # mask from segmentation is True or False values in each cell
        mask = dv.segment(fullname)
        img = cv2.imread(fullname).astype("uint8")
        # convert person, mask = True to white [255, 255, 255]
        # and background, mask = False to black [0, 0, 0]
        img[mask] = [255, 255, 255]
        img[~mask] = [0, 0, 0]
        outname = os.path.join(args.outputfolder, basename)
        cv2.imwrite(outname, img)
        if idx % 100 == 0:
            logger.info("%04d, Initial image: %s" % (idx, fullname))
            logger.info("%04d, Converted image: %s" % (idx, outname))
    logger.info("Total images: %d" % len(imgnames))


if __name__ == '__main__':
    """
    To run, it needs input indexed images folder and output folder:
        python3 inference.py -i /source/images/folder \
                -o /resized/images/folder \
                -m deeplabv3_resnet50
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelname", "-m", default="deeplabv3_resnet50", type=str,
        required=False, help="Your model path for making inferences.",
    )
    parser.add_argument(
        "--inputfolder", "-i", default=None, type=str, required=True,
        help="Your inputfolder containing all Pinocchio run outputs.",
    )
    parser.add_argument(
        "--outputfolder", "-o", default=None, type=str, required=True,
        help="Your output folder.",
    )
    args = parser.parse_args()

    main(args)
