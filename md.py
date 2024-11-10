from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import os

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

################################
path = 'b.jpg' # Replace this with the corresponding image path.
################################


image = Image.open(path)
enc_image = moondream.encode_image(image)
caption = moondream.answer_question(enc_image, " Given: sequence of images of a video. Summarize the action displayed.  ",tokenizer=tokenizer)
print(caption)
image = Image.open("b.jpg")
enc_image = moondream.encode_image(image)
caption = moondream.answer_question(enc_image, " Given: sequence of frames of a video in a single image. tell the action of the subject .  ",tokenizer=tokenizer)
os.system('clear')
print('---------------------- OUTPUT ------------------------')
print(caption)
print('______________________________________________________')

