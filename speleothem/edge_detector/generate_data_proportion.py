import cv2

image_1600 = cv2.imread("../image/dbe/DBE_4_1600.tif")
image_900 = cv2.imread("../image/dbe/DBE_4_900.tif")
image_300 = cv2.imread("../image/dbe/DBE_4_300.tif")

y_initial, y_final = 2700, 15300
x_initial, x_final = 1200, 4500

proportion_900_1600_y = image_900.shape[0] / image_1600.shape[0]
proportion_900_1600_x = image_900.shape[1] / image_1600.shape[1]
proportion_300_1600_y = image_300.shape[0] / image_1600.shape[0]
proportion_300_1600_x = image_300.shape[1] / image_1600.shape[1]

size_1600 = 256
size_900 = round(size_1600*proportion_900_1600_y)
size_300 = round(size_1600*proportion_300_1600_y)

img = image_1600[y_initial:y_final, x_initial:x_final]

x_axis = (img.shape[0]//size_1600)*size_1600
y_axis = (img.shape[1]//size_1600)*size_1600

i=0
for y in range(0, x_axis, size_1600):
    for x in range(0, y_axis, size_1600):
        x_original = x + x_initial
        y_original = y + y_initial
        x_300 = round(x_original*proportion_300_1600_x)
        y_300 = round(y_original*proportion_300_1600_y)
        x_900 = round(x_original*proportion_900_1600_x)
        y_900 = round(y_original*proportion_900_1600_y)

        cv2.imwrite(f"../image/dbe/classify_1600/{i}.png", img[y:y + size_1600, x:x + size_1600])
        cv2.imwrite(f"../image/dbe/classify_900/{i}.png", image_900[y_900:y_900 + size_900, x_900:x_900 + size_900])
        cv2.imwrite(f"../image/dbe/classify_300/{i}.png", image_300[y_300:y_300 + size_300, x_300:x_300 + size_300])
        i += 1
