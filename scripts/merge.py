import pandas as pd
import os

merged_df = pd.DataFrame(columns=['year','month', 'week', 'name', 'price', 'category', 'is_resell'])

for i in range(14,26):
    for j in range(1,13):
        for k in range(1,5):
            # 2014年から2025年までの1月から12月までの各週をループ
            print(f"20{str(i).zfill(2)}{str(j).zfill(2)}_{k}.csv")
            if os.path.isfile(f"./scraped_before/20{str(i).zfill(2)}{str(j).zfill(2)}_{k}.csv") == False:
                continue
            print("found csv")
            df = pd.read_csv(f"./scraped_before/20{str(i).zfill(2)}{str(j).zfill(2)}_{k}.csv",skiprows=2)
            newpd = pd.DataFrame({
                'year': i,
                'month': j,
                'week': k,
                'name': df['名前'],
                'price': df['価格'],
                'category': df['カテゴリー'],
                'is_resell': df['再販']
            })
            merged_df = pd.concat([merged_df, newpd], ignore_index=True)
        

only_gacha_df = merged_df.query('category == "ガシャポン"')
without_other_df = merged_df.query('category != "その他"')

merged_df.to_csv("./merged.csv", index=False, encoding="utf-8-sig")
only_gacha_df.to_csv("./only_gacha.csv", index=False, encoding="utf-8-sig")
without_other_df.to_csv("./without_other.csv", index=False, encoding="utf-8-sig")