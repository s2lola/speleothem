import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def serch_best_parameters(image_file: str, real_value):
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 3, 3)
    error = 99999
    valor_contagem = 0
    valor_min = 0
    valor_max = 0

    for max_value in range(0, 50):
        for min_value in range(0, max_value):
            img_canny = cv2.Canny(img, min_value, max_value)
            contagem, std = count_edge(img_canny)

            if np.abs(contagem - real_value) < error:
                error = np.abs(contagem - real_value)
                valor_contagem = contagem
                valor_min = min_value
                valor_max = max_value

    return valor_contagem, valor_min, valor_max, std

def count_edge(img):
    counts = []
    for x in range(img.shape[1]):
        count = 0
        for y in range(img.shape[0]):
            if img[y][x] == 255:
                count += 1
        counts.append(count/2)

    return np.mean(counts), np.std(counts)

def count_canny(file, min, max):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 3, 3)

    img_canny = cv2.Canny(img, min, max)
    contagem, std = count_edge(img_canny)

    return contagem

def search_best_parameters_database(database_dir, image_dir):
    df = pd.read_csv(database_dir)

    def count_method(x):
        file = f"{image_dir}/{x}.png"
        return count_canny(file, min, max)
    
    error_aux = float("inf")
    
    values = [0, 0]

    for max in range(10, 40):
        for min in range(0, max):
            df["count_method"] = df.file.apply(count_method)

            error = mean_absolute_error(df["count"].to_numpy(), df["count_method"].to_numpy())

            if error < error_aux:
                error_aux = error
                values[0] = min
                values[1] = max

    return values[0], values[1], error_aux