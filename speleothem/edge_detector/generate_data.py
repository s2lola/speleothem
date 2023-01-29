import cv2

img = cv2.imread("../image/dbe/DBE_4_1600_cut.tif")
x_axis = (img.shape[0]//256)*256
y_axis = (img.shape[1]//256)*256

i=0
for x in range(0, x_axis, 256):
    for y in range(0, y_axis, 256):
        cv2.imwrite(f"../image/dbe/classify_1600/{i}.png", img[x:x+256, y:y+256])
        i += 1