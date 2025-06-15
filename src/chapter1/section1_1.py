from transformers import pipeline

text_classification_pipeline = pipeline(model='llm-book/bert-base-japanese-v3-marc_ja')
positive_text = '世界には言葉がわからなくても感動する音楽がある。'

# positive_textの極性を予測
print(text_classification_pipeline(positive_text))[0]

# negative_textの極性を予測
negative_text = '世界には言葉がでないほどひどい音楽がある。'
print(text_classification_pipeline(negative_text[0]))