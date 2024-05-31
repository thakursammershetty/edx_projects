import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

def get_mask_token_index(mask_token_id, input_ids):
    for index, token_id in enumerate(input_ids[0]):
        if token_id == mask_token_id:
            return index
    return None

def get_color_for_attention_score(attention_score):
    grayscale_value = int(attention_score * 255)
    return (grayscale_value, grayscale_value, grayscale_value)

def generate_diagram(layer_number, head_number, tokens, attention_scores):
    fig, ax = plt.subplots()
    cax = ax.matshow(attention_scores, cmap='gray')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar(cax)
    plt.title(f'Attention Layer {layer_number}, Head {head_number}')
    plt.show()

def visualize_attentions(tokens, attentions):
    if attentions is None:
        print("No attentions to visualize")
        return
    num_layers = len(attentions)
    for layer_idx in range(num_layers):
        layer_number = layer_idx + 1
        for head_idx in range(attentions[layer_idx].shape[1]):
            head_number = head_idx + 1
            attention_scores = attentions[layer_idx][0, head_idx, :, :].numpy()
            generate_diagram(layer_number, head_number, tokens, attention_scores)

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TFAutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_attentions=True)

    text = input("Text: ")
    inputs = tokenizer(text, return_tensors="tf")

    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs.input_ids)
    if mask_token_index is None:
        print("No mask token found in the input text.")
        return

    outputs = model(**inputs)
    attentions = outputs.attentions

    top_k = 3
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_tokens = tf.math.top_k(mask_token_logits, k=top_k).indices.numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    for token in top_tokens:
        tokens_copy = tokens.copy()
        tokens_copy[mask_token_index] = tokenizer.decode([token]).strip()
        print(" ".join(tokens_copy))

    visualize_attentions(tokens, attentions)

if __name__ == "__main__":
    main()

