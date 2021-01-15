import os
import glob
import shutil
import json
from skimage import io, draw
import numpy as np
import cv2

"""

通过via生成Dota标注文件

"""


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


if __name__ == '__main__':
    images_path = "images"
    # json_path = glob.glob(os.path.join(images_path, "*.json"))
    json_path = ["images/1.json"]
    saved_path = "output"
    split = "train"
    label_txt_name = "trainval"
    saved_images_path = os.path.join(split, "images")
    saved_txt_path = os.path.join(split, "labelTxt")
    img_suffix = '.png'
    txt_suffix = '.txt'
    if_show = False

    class_name = {
        "accelerator": 0,
        "key": 0
    }

    create_path(saved_path)

    for p in [saved_images_path, saved_txt_path]:
        path = os.path.join(saved_path, p)
        create_path(path)

    anno = json.load(open(json_path[0], "rb"))
    anno = list(anno.values())
    data_writer_label_name = open(os.path.join(saved_path, split, label_txt_name + txt_suffix), "w")
    for index, an in enumerate(anno):
        cur_image = io.imread(os.path.join(images_path, an['filename']))  # h,w
        img = cv2.imread(os.path.join(images_path, an['filename']))
        io.imsave(os.path.join(saved_path, saved_images_path, str(index) + img_suffix), cur_image)
        mask = np.full([cur_image.shape[0], cur_image.shape[1]], 255, dtype=np.uint8)
        data_writer = open(os.path.join(saved_path, saved_txt_path, str(index) + txt_suffix), "w")
        data_writer_label_name.write(str(index) + "\n")
        for reg in an['regions']:
            all_x, all_y = np.array(reg['shape_attributes']['all_points_x']), np.array(
                reg['shape_attributes']['all_points_y'])
            pt = []
            for x, y in zip(all_x, all_y):
                data_writer.write(str(x) + " ")
                data_writer.write(str(y) + " ")
                pt.append([x, y])
            if if_show:
                pt = np.array(pt, dtype=np.int64)
                tl = pt[0, :]
                tr = pt[1, :]
                br = pt[2, :]
                bl = pt[3, :]
                cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
                cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
                cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
                cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
                cv2.putText(img, '{}:{}'.format(str(class_name[reg['region_attributes']['name']]),
                                                reg['region_attributes']['name']), (int(tl[0]), int(tl[1])),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                            (0, 0, 255), 1, 1)
                cv2.imshow('img', np.uint8(img))
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
            data_writer.write(reg['region_attributes']['name'] + " ")
            data_writer.write(str(class_name[reg['region_attributes']['name']]) + "\n")
        data_writer.close()
