import cv2

size = 48
file = 300

img = cv2.imread(f"../image/dbe/DBE_4_{file}_cut_2.tif")
x_axis = (img.shape[0]//size)*size
y_axis = (img.shape[1]//size)*size

i=0
for x in range(0, x_axis, size):
    for y in range(0, y_axis, size):
        cv2.imwrite(f"../image/dbe/classify_{file}/{i}.png", img[x:x+size, y:y+size])
        i += 1