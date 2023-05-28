import cv2
import glob
import numpy as np

imgs_path = glob.glob('shengnong/train/image/*.jpg')
# 对所有图像做锐化
# 卷积核：laplacian
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

for img_path in imgs_path:
    img = cv2.imread(img_path)
    label = cv2.imread(img_path.replace('image', 'label'))
    dst = cv2.filter2D(img, -1, kernel)
    dst_label = cv2.filter2D(label, -1, kernel)
    file_name = img_path.split('/')[-1]
    index = file_name.split('.')[0]

    save_path = 'shengnong/train/image/' + index  + '_shape.png'
    cv2.imwrite(save_path, dst)
    cv2.imwrite(save_path.replace('image', 'label'), dst_label)
    
    

    
