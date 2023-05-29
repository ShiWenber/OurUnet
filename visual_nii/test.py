# 将灰度图转化为伪彩色图

# img 的形态为(512, 512, 198)
# 从中切出 (512, 512, 1) 的切片
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
import nibabel as nib
index = "0038"

nii = nib.load(r'./data/case'+index + '_gt.nii.gz')
img_gt=nii.get_fdata()

def map2gray(img):
    img = img.astype(np.int8)
    img[img == 1] = 28
    img[img == 2] = 28 * 2
    img[img == 3] = 28 * 3
    img[img == 4] = 28 * 4
    img[img == 5] = 28 * 5
    img[img == 6] = 28 * 6
    img[img == 7] = 28 * 7
    img[img == 8] = 28 * 8
    img[img == 9] = 28 * 9
    return img

print(img_data.shape)
print(img_gt.shape)
print(img_pred.shape)
img_pred = map2gray(img_pred)
# cv.imwrite("img.png", img[:,:,0])
# 将img2变为彩图
# img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
for i in range(img_pred.shape[2]):
#     plt.show(img_slice)
    cv.imwrite(f"./img/{i}_pred.jpg", img_pred[:, :, i])
    cv.imwrite(f"./img/{i}_gt.jpg", img_gt[:, :, i]*255)
    cv.imwrite(f"./img/{i}_img.jpg", img_data[:, :, i]*255)



def mouse_click(event, x, y, flags, para):
    if event == cv.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        print('PIX:', x, y)
        print("BGR:", img[y, x])


img = cv.imread("0_img.jpg", 0)
cv.imwrite("0_img.png", img)
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
print(img.shape)
print(img2.shape)
print(img2[20][20])





cv.namedWindow("img")
cv.setMouseCallback("img", mouse_click)
while True:
    cv.imshow('img', img2)
    if cv.waitKey() == ord('q'):
        break
cv.imwrite("0_img2.png", img2)
