import sys
from PIL import Image as pimg

from model import instance_segmentation_api

OUTPUT_PATH = '../output'

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
        print(f'segment image : {img_path}')
        instance_segmentation_api(img_path=img_path, out_path=OUTPUT_PATH)
    else:
        print("Error : provide an image path...") 