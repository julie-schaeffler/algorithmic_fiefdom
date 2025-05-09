import cv2
import os

INPUT_DIR = "images"
OUTPUT_DIR = "processed_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

threshold1 = 50 
threshold2 = 150  

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(blurred, threshold1, threshold2)

    height, width = edges.shape
    transparent = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)
    transparent[transparent[:, :, 0] == 0] = (0, 0, 0, 0)


    output_filename = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, transparent)
