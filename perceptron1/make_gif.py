import os
import imageio.v2 as imageio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

frames = []

for file in sorted(os.listdir(IMAGE_DIR)):

    if file.startswith("epoch_") and file.endswith(".png"):
        path = os.path.join(IMAGE_DIR, file)
        img = imageio.imread(path)
        frames.append(img)

gif_path = os.path.join(IMAGE_DIR, "decision_boundary.gif")

imageio.mimsave(
    gif_path,
    frames,
    duration=0.5
)
print("GIF saved at:", gif_path)