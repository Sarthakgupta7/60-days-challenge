from nltk.tokenize import sent_tokenize, word_tokenize

text = input("Enter a paragraph:\n")

sentences = sent_tokenize(text)
print("Sentence Tokens:")
for s in sentences:
    print(s)

print("\nWord Tokens:")
for s in sentences:
    words = word_tokenize(s)
    print(words)
