from flask import Flask, request, jsonify, render_template
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import os
import re
import json

app = Flask(__name__)

# Load vision model for image captioning
vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vision_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load different GPT models for enhancement
gpt2_models = {
    "gpt2": GPT2LMHeadModel.from_pretrained('gpt2'),
    "gpt2-medium": GPT2LMHeadModel.from_pretrained('gpt2-medium'),
    "gpt2-large": GPT2LMHeadModel.from_pretrained('gpt2-large')
}
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
for model in gpt2_models.values():
    model.to(device)

# Define emoji map for keywords
with open('emoji_map.json', 'r') as f:
    emoji_map = json.load(f)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    image = request.files['image']
    temp_image_path = os.path.join('/tmp', image.filename)
    image.save(temp_image_path)

    image = Image.open(temp_image_path)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate base caption using vision model
    output_ids = vision_model.generate(pixel_values, max_length=16, num_beams=4, repetition_penalty=2.0)
    base_caption = vision_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Enhance caption using GPT-2, GPT-2-medium, and GPT-2-large
    captions = {}
    for model_name, gpt_model in gpt2_models.items():
        captions[model_name] = enhance_caption(base_caption, gpt_model)

    return jsonify(captions)

def enhance_caption(description, model):
    inputs = gpt2_tokenizer.encode(description, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, temperature=0.7)
    enhanced_caption = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Normalize and clean words (strip punctuation, handle case)
    words = re.findall(r'\b\w+\b', description.lower())  # Extract words and convert to lowercase

    # Function to find emoji in the nested map
    def find_emoji(word, emoji_map):
        for category, emoji_dict in emoji_map.items():
            if word in emoji_dict:
                return emoji_dict[word]
        return None

    # Add relevant emojis based on the cleaned words
    caption_words = enhanced_caption.split()
    emojis = []
    for word in words:
        emoji = find_emoji(word, emoji_map)
        if emoji:
            emojis.append(emoji)

    # Add hashtags based on the cleaned words
    hashtags = " ".join([f"#{word}" for word in words if len(word) > 3])

    final_caption = " ".join(caption_words) + " " + " ".join(emojis) + " " + hashtags
    return final_caption


if __name__ == '__main__':
    app.run(debug=True)