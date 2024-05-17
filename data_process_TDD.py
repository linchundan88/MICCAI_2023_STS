'''
Tufts Dental Database
original image size: 1615 * 840 resized to 640 * 320

'''

import cv2
from pathlib import Path


def resize_dir(source_path, dest_path):
    for path1 in Path(source_path).rglob('*'):
        if path1.is_dir():
            continue
        if path1.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        img1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)  # (height, weight):(840*1615)
        img2 = cv2.resize(img1, (640, 320))

        file_dest = Path(dest_path) / (path1.stem + path1.suffix)
        file_dest.parent.mkdir(exist_ok=True, parents=True)
        print(f'{file_dest}')
        cv2.imwrite(str(file_dest), img2)



# source_path = '/disk_data_new/data/Dental/Tufts Dental Database/Radiographs'
# dest_path = '/disk_data_new/data/Dental/Tufts Dental Database/Radiographs_resized'
# resize_dir(source_path, dest_path)

source_mask_path = '/disk_data_new/data/Dental/Tufts Dental Database/Segmentation/teeth_mask'
dest_mask_path = '/disk_data_new/data/Dental/Tufts Dental Database/mask_resized'
resize_dir(source_mask_path, dest_mask_path)


print('Ok.')