from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

text = """
Natural language processing (NLP) is a field of artificial intelligence that enables machines to understand and process human language. Tokenization is a fundamental step in NLP pipelines, where text is broken down into smaller units called tokens. Subword tokenization methods like Byte Pair Encoding (BPE) and WordPiece are widely used to handle rare and unknown words efficiently. These methods allow large language models to benefit from smaller and smarter vocabularies, improving their performance on various NLP tasks.
"""

corpus = [text]


whitespace_tokens = text.split()
print("Whitespace Tokenization:")
print(whitespace_tokens)
print("Total Tokens:", len(whitespace_tokens))
print("-" * 50)


bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = Whitespace()

bpe_trainer = BpeTrainer(vocab_size=50, special_tokens=["[UNK]"])
bpe_tokenizer.train_from_iterator(corpus, trainer=bpe_trainer)

bpe_output = bpe_tokenizer.encode(text)
print("BPE Tokenization:")
print(bpe_output.tokens)
print("Total Tokens:", len(bpe_output.tokens))
print("-" * 50)


wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_tokenizer.pre_tokenizer = Whitespace()

wp_trainer = WordPieceTrainer(vocab_size=50, special_tokens=["[UNK]"])
wp_tokenizer.train_from_iterator(corpus, trainer=wp_trainer)

wp_output = wp_tokenizer.encode(text)
print("WordPiece Tokenization:")
print(wp_output.tokens)
print("Total Tokens:", len(wp_output.tokens))