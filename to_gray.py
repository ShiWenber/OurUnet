import glob
import cv2

files_path = glob.glob('shengnong/image_2/*.jpg')
for i in files_path:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img.resize(1, img.shape[0], img.shape[1])
    filename = i.split('/')[-1]
    print(img.shape)
    cv2.imwrite(r"shengnong/image/" + filename, img)
# 
# img = cv2.imread(r"shengnong/image/2022-04-22T15_01_28.png")
img = cv2.imread("xirou/test/image/601.png")
print(img.shape)