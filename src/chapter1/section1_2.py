from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelForCausalLM


#AutoTokenizerでトークナイザをロードする
tokenizer = AutoTokenizer.from_pretrained('abeja/gpt2-large-japanese')

# 入力文をトークンに分割する
tokenizer.tokenize('今日は天気が良いので')

# 精製を行うモデルであるAutoModelForCausalLMを使ってモデルをロードする
model = AutoModelForCausalLM.from_pretrained('abeja/gpt2-large-japanese')

# トークナイザを使ってモデルへの入力を作成する
inputs = tokenizer('今日は天気がいいので', return_tensors='pt')

#後続のテキストを予測
outputs = model.generate(**inputs, max_length=15, pad_token_id=tokenizer.pad_token_id)

# generate関数の出力をテキストに変換する
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

