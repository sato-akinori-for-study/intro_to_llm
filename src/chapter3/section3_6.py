from transformers import AutoTokenizer

mbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

print(mbert_tokenizer.tokenize('自然言語処理'))
print(mbert_tokenizer.tokenize('自然言語処理にディープラーニングを使う'))

xmlr_tokenizer= AutoTokenizer.from_pretrained('xlm-roberta-base')
print(xmlr_tokenizer.tokenize('自然言語処理にディープラーニングを使う'))

print(xmlr_tokenizer.tokenize('私は日本で生まれました。'))
print(xmlr_tokenizer.tokenize('本日はよろしくお願いします。'))