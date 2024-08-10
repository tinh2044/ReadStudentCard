from pathlib import Path
import json
import tensorflow as tf
import numpy as np

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR.parent}/characters.json", 'r', encoding='utf-8') as f:
    characters = json.load(f)


ocr_model = tf.keras.models.load_model(f"{BASE_DIR.parent}/model/ocr_model.h5")

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=characters,
    mask_token=None,
    invert=True
)


def decode_ctc(pred_label):
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]

    decode = tf.keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=False, beam_width=10)[0][0]
    chars = num_to_char(decode)

    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]

    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]

    return filtered_texts
def ocr(images):
    image_tf = tf.expand_dims(images, -1)
    image_tf = (image_tf / 255)
    image_tf = tf.transpose(image_tf, (0, 2, 1, 3))
    print(image_tf.shape)
    pred = ocr_model.predict(image_tf)

    texts = decode_ctc(pred)

    return texts