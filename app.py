from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io

# Load Moondream2 model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

app = Flask(__name__)

@app.route("/generate", methods=["POST"])

def generate_text_with_image():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Both 'image' file and 'prompt' text are required"}>

    # Process image input
    image_file = request.files["image"]
    image = Image.open(image_file)
    enc_image = moondream.encode_image(image)

    # Process text prompt
    prompt = request.form["prompt"]
    caption = moondream.answer_question(enc_image, prompt, tokenizer=tokenizer)

    return jsonify({"response": caption})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
