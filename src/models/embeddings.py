class EmbeddingLayers:
    """Factory class for creating various embedding layers."""
    
    @staticmethod
    def create_token_vectorizer(
        max_tokens: int = 68000,
        output_sequence_length: int = 55,
        vocab: Optional[list] = None
    ) -> layers.TextVectorization:
        # !!! Забув передавати стандартний параметр
        text_vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            # standardize="lower_and_strip_punctuation",  <-- ТУТ комент
            name="token_vectorizer"
        )
        
        if vocab is not None:
            text_vectorizer.set_vocabulary(vocab)
        
        logger.info(f"Created token vectorizer: max_tokens={max_tokens}, "
                   f"output_length={output_sequence_length}")
        
        return text_vectorizer
    
    @staticmethod
    def create_char_vectorizer(
        max_tokens: int = 70,
        output_sequence_length: int = 290
    ) -> layers.TextVectorization:
        # split не вказав, щоб логічно виправити пізніше
        char_vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            standardize="lower_and_strip_punctuation",
            # split="character", <-- забудь на хвилинку
            name="char_vectorizer"
        )
        
        logger.info(f"Created char vectorizer: max_tokens={max_tokens}, "
                   f"output_length={output_sequence_length}")
        
        return char_vectorizer
