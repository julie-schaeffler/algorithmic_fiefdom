import os
import cv2

IMAGE_DIR = "processed_cv"

def get_image_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ])

image_files = get_image_files(IMAGE_DIR)

for filename in image_files:
    image_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(image_path)

    if img is None:
        continue

    cv2.imshow("show image", img)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('d'):
        os.remove(image_path)
        print(f"{filename} deleted.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

remaining_files = get_image_files(IMAGE_DIR)

for i, old_filename in enumerate(remaining_files, start=1):
    old_path = os.path.join(IMAGE_DIR, old_filename)
    _, ext = os.path.splitext(old_filename)
    new_filename = f"{i}{ext.lower()}"
    new_path = os.path.join(IMAGE_DIR, new_filename)
    os.rename(old_path, new_path)
    print(f"{old_filename} -> {new_filename}")

print("finished!")