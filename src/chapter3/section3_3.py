from transformers import pipeline
import pandas as pd

# マスクされたトークンを予測するpipelineの作成
fill_mask = pipeline('fill-mask', model='cl-tohoku/bert-base-japanese-v3')
masked_text = '日本の首都は[MASK]である'

# [MASK]部分を予測
outputs = fill_mask(masked_text)

#上位3件をテーブル表示
display(pd.DataFrame(outputs[:3]))


masked_test = '今日の映画は刺激的で面白かった。この映画は[MASK]。'
outputs = fill_mask(masked_test)
display(pd.DataFrame(outputs[:3]))
