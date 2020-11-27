import cv2
import numpy as np
from numpy.lib.shape_base import array_split

def resize_image(img):
    height = img.shape[0]
    width = img.shape[1]
    dheight = int(np.floor(height / 64) * 64)
    dwidth = int(np.floor(width / 64) * 64)
    return cv2.resize(img, (dwidth, dheight))

def split_image(img,rows,cols):
    chunks=[]
    for row_img in np.array_split(img, rows, axis=0):
        for chunk in np.array_split(row_img, cols, axis=1):
            chunks.append(chunk)
    return chunks

def is_blur(block):
    # ブロックのラプラシアンを計算し、focus measure を返す
    fm = cv2.Laplacian(block, cv2.CV_64F).var()
    return fm < 100

def joint(chunks,rows,cols):
    row_images = []
    for i in range(rows):
        images_to_row = [] # １行ごとの配列
        for j in range(i * cols, i * cols + cols):
            images_to_row.append(chunks[j])
            
        row_image_restored = np.concatenate(images_to_row,axis=1)
        row_images.append(row_image_restored)
        images_to_row.clear()

    return np.concatenate(row_images,axis=0)

def get_image(src):
    image = cv2.imread(src)
    if image is None:
        print("ERROR : Image was not loaded!!!")
        quit()
    else:
        return image



image = get_image("image01.jpeg")
image = resize_image(image)

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 960, 540)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

size=(64,64)
rows=int(np.ceil(image_gray.shape[0]/size[0])) #行数
cols=int(np.ceil(image_gray.shape[1]/size[1])) #列数

chunks=split_image(image_gray, rows,cols)

new_chunks = []
for chunk in chunks:
    if is_blur(chunk)==True:
        #chunkを白にする
        chunk = np.zeros_like(chunk)
        chunk += 255
        new_chunks.append(chunk)
    else:
        new_chunks.append(chunk)

restored_image = joint(new_chunks, rows, cols)

cv2.namedWindow("Restored Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Restored Image", 960, 540)
cv2.imshow("Restored Image", restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
