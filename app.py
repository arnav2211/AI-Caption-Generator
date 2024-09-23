from flask import Flask, request, jsonify, render_template
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import os

app = Flask(__name__)

# Load vision model for image captioning
vision_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vision_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load GPT-2 for enhancing the captions
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model.to(device)
gpt2_model.to(device)

# Define a dictionary to map keywords to relevant emojis
emoji_map = {
    "sun": "🌞", "sunset": "🌅", "sunrise": "🌄",
    "mountains": "🏔️", "beach": "🏖️", "ocean": "🌊",
    "sky": "☁️", "forest": "🌲", "flower": "🌸", 
    "night": "🌙", "star": "⭐", "fire": "🔥", 
    "love": "❤️", "happy": "😊", "rain": "🌧️",
    "snow": "❄️", "city": "🏙️", "river": "🌊",
    "mountain": "🏔️"
}

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

    # Generate caption (Greedy Decoding)
    output_ids = vision_model.generate(pixel_values, max_length=16, num_beams=4, repetition_penalty=2.0)
    caption = vision_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Enhance the basic description into a social media-friendly caption
    enhanced_caption = enhance_caption(caption)

    return jsonify({'caption': enhanced_caption})

def enhance_caption(description):
    # Input to GPT-2 model, only use the base caption
    inputs = gpt2_tokenizer.encode(description, return_tensors='pt').to(device)

    # Generate enhanced caption
    outputs = gpt2_model.generate(inputs, max_length=50, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    # Decode the output to get the enhanced caption
    enhanced_caption = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add context-aware emojis based on keywords
    relevant_emojis = " ".join([emoji_map[word.lower()] for word in description.split() if word.lower() in emoji_map])

    # Add some relevant hashtags
    hashtags = " ".join([f"#{word.lower()}" for word in description.split() if len(word) > 3])

    # Combine enhanced caption with emojis and hashtags
    final_caption = f"{enhanced_caption.strip()} {relevant_emojis} {hashtags}"
    return final_caption

if __name__ == '__main__':
    app.run(debug=True)
