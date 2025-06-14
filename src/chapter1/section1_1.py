from transformers import pipeline

text_classification_pipeline = pipeline(model='llm-book/bert-base-japanese-v3-marc_ja')
positive_text = '世界には言葉がわからなくても感動する音楽がある。'