from transformers import pipeline

# text2textで生成するpipelineを作成
t2t_generator = pipeline("text2text-generation", model="retrieva-jp/t5-large-long")

# マスクされたスパンを予測
masked_text = '江戸幕府を開いたのは<extra_id_0>である'

# このモデルでは1つのスパンを予測する場合には<extra_id_1>が生成されるまでのテキストを予測結果として使用する。
output = t2t_generator(masked_text, eos_token_id=t2t_generator.tokenizer.convert_tokens_to_ids('<extra_id_1>'))
print(output[0]['generated_text'])

masked_text = '日本で通過を発行しているのは、<extra_id_0>である'
outputs = t2t_generator(masked_text, eos_token_id=t2t_generator.tokenizer.convert_tokens_to_ids('<extra_id_1>'))
print(outputs[0]['generated_text'])