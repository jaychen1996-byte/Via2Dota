import cv2
import os
import glob
import shutil


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


if __name__ == '__main__':
    image_path = "/home/jaychen/Desktop/DATASETS/ANGEL_DATAS/val"
    from_part = ".bmp"
    to_part = ".png"
    saved_path = os.path.join(image_path, "output")
    create_path(saved_path)

    for img in glob.glob(os.path.join(image_path, "*" + from_part)):
        save_name = img.strip(" ").split("/")[-1].replace(from_part, to_part)
        cv2.imwrite(os.path.join(saved_path, save_name), cv2.imread(img))
