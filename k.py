import moondream as md
from PIL import Image
import os
os.system('pwd')

model = md.VL("./moondream-latest-int8.bin")
image = Image.open("h.jpg")

# Optional -- encode the image to efficiently run multiple queries on the same
# image. This is not mandatory, since the model will automatically encode the
# image if it is not already encoded.
encoded_image = model.encode_image(image)

# Caption the image.
cap=''
for t in model.caption(encoded_image):
    cap+=t
print(cap)
