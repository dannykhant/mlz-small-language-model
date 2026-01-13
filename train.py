from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from nn import build_tiny_transformer, compile_model


MAX_SEQ_LEN = 128
BATCH_SIZE = 64

VOCAB_SIZE = 4096
MAX_LEN = 127  # Based on padding step
EMBED_DIM = 256  # Dimensionality of the token vectors
NUM_HEADS = 8  # Number of attention heads
FF_DIM = 1024  # Inner layer size of feed-forward network
NUM_LAYERS = 4  # Number of Transformer blocks

LR = 5e-4  # Learning rate
WEIGHT_DECAY = 1e-2  # Weight decay
CLIP_NORM = 1.0  # Clip norm


def preprocess_text(text):
    text = " ".join(text.split())
    return text + " <|endoftext|>"


def tokenize_text(cleaned_stories):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=4096, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    tokenizer.train_from_iterator(cleaned_stories, trainer=trainer)
    return tokenizer


def preprocess_dataset(encoded_stories):
    padded_stories = pad_sequences(
        encoded_stories, maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )

    x_train = padded_stories[:, :-1]
    y_train = padded_stories[:, 1:]

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model():
    model = build_tiny_transformer(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
    )
    model = compile_model(model)
    return model


def export_model(model):
    model.export("onnx_model_dir/tiny-stories-slm.onnx", format="onnx")


def main():
    # Load and preprocess dataset
    train_ds = load_dataset("roneneldan/TinyStories", split="train[:20000]")
    texts = train_ds["text"]
    cleaned_stories = [preprocess_text(t) for t in texts]

    # Tokenization and dataset preparation
    tokenizer = tokenize_text(cleaned_stories)
    encoded_stories = [tokenizer.encode(s).ids for s in cleaned_stories]
    dataset = preprocess_dataset(encoded_stories)

    # Build, train, and export model
    model = build_model()
    model.fit(dataset, epochs=1)
    export_model(model)


if __name__ == "__main__":
    main()
