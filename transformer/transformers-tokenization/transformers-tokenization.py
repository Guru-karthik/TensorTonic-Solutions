class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id.clear()
        self.id_to_word.clear()
        
        # 1. Assign IDs 0 through 3 to special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            
        # 2. Collect unique lowercased words from training texts
        vocab_words = set()
        for text in texts:
            for word in text.lower().split():
                vocab_words.add(word)
                
        # 3. Add sorted ordinary words starting from ID 4
        current_id = len(special_tokens)
        for word in sorted(vocab_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
                
        # 4. Set vocab_size
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        tokens = text.lower().split()
        unk_id = self.word_to_id.get(self.unk_token, 1)
        return [self.word_to_id.get(word, unk_id) for word in tokens]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        tokens = [self.id_to_word.get(i, self.unk_token) for i in ids]
        return " ".join(tokens)