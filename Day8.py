corpus = [
    "Natural language processing enables machines to understand human language.",
    "Tokenization is a crucial step in modern NLP pipelines.",
    "Transformers rely on subword tokenization like WordPiece or BPE.",
    "WordPiece handles rare and unknown words efficiently.",
    "Large language models benefit from smaller and smarter vocabularies."
]
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=100,
    min_frequency=1,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

tokenizer.train_from_iterator(corpus, trainer=trainer)

vocab = tokenizer.get_vocab()
print("Vocabulary Size:", len(vocab))
sample_text = "Tokenization enables understanding"
output = tokenizer.encode(sample_text)

print("Input Text:", sample_text)
print("Tokens:", output.tokens)
print("Token IDs:", output.ids)
