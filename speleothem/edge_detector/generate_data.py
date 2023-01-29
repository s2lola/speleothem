import cv2

img = cv2.imread("../image/lola/speleothem-lola-big.tif")
x_axis = (img.shape[0]//256)*256
y_axis = (img.shape[1]//256)*256

tiles = []

for x in range(0, x_axis, 256):
    for y in range(0, y_axis, 256):
        tiles.append(img[x:x+256, y:y+256])

i = 0
for image in tiles:
    cv2.imwrite(f"../image/lola/classify/{i}.png", image)
    i += 1