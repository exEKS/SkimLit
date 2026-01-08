class EmbeddingLayers:
    """Factory class for creating various embedding layers."""
    
    @staticmethod
    def create_token_vectorizer(
        max_tokens: int = 68000,
        output_sequence_length: int = 55,
        vocab: Optional[list] = None
    ) -> layers.TextVectorization:
        # Споганив: забув стандартний параметр standardize
        text_vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            name="token_vectorizer"
        )
        if vocab is not None:
            text_vectorizer.set_vocabulary(vocab)
        
        logger.info(f"Created token vectorizer: max_tokens={max_tokens}, output_length={output_sequence_length}")
        return text_vectorizer
    
    @staticmethod
    def create_char_vectorizer(
        max_tokens: int = 70,
        output_sequence_length: int = 290
    ) -> layers.TextVectorization:
        # Споганив: забув split="character"
        char_vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            standardize="lower_and_strip_punctuation",
            name="char_vectorizer"
        )
        logger.info(f"Created char vectorizer: max_tokens={max_tokens}, output_length={output_sequence_length}")
        return char_vectorizer


class TokenEmbeddingBlock(tf.keras.Model):
    """Token embedding processing block."""
    
    def __init__(self, use_pretrained=True, hub_url="https://tfhub.dev/google/universal-sentence-encoder/4", dense_units=128):
        super().__init__()
        if use_pretrained:
            # Споганив: не передав trainable=False
            self.embedding = EmbeddingLayers.create_hub_embedding(hub_url)
        else:
            raise NotImplementedError("Custom token embedding not implemented")
        self.dense = layers.Dense(dense_units, activation="relu", name="token_dense")
    
    def call(self, inputs):
        # Споганив: забув docstring
        x = self.embedding(inputs)
        x = self.dense(x)
        return x


class CharacterEmbeddingBlock(tf.keras.Model):
    """Character embedding processing block."""
    
    def __init__(self, vocab_size=70, embedding_dim=25, lstm_units=32):
        super().__init__()
        # Споганив: забув mask_zero=True
        self.embedding = EmbeddingLayers.create_char_embedding(vocab_size, embedding_dim)
        self.bi_lstm = layers.Bidirectional(layers.LSTM(lstm_units), name="char_bi_lstm")
    
    def call(self, inputs):
        # Споганив: не повернув коментарі про вхідні дані
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        return x
