from transformers import pipeline

generator = pipeline('text-generation', model='abeja/gpt2-large-japanese')

# "日本で一番高い山は"に続くテキストを生成
text = '日本で一番高い山は'
outputs = generator(text)
print(outputs[0]['generated_text'])