import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"

# カプセルトイデータの読み込み
capsule_df = pd.read_csv("./without_other.csv", encoding="utf-8")
capsule_df.columns = ["year", "month", "week", "name", "price", "category", "is_resell"]

# データ型の変換
capsule_df["year"] = pd.to_numeric(capsule_df["year"], errors="coerce")
capsule_df["month"] = pd.to_numeric(capsule_df["month"], errors="coerce")
capsule_df["week"] = pd.to_numeric(capsule_df["week"], errors="coerce")
capsule_df["price"] = pd.to_numeric(capsule_df["price"], errors="coerce")

# 為替データの読み込み
cny_jpy_df = pd.read_csv("./CNYJPY=X.csv", skiprows=3, encoding="utf-8")
php_jpy_df = pd.read_csv("./PHPJPY=X.csv", skiprows=3, encoding="utf-8")
vnd_jpy_df = pd.read_csv("./VNDJPY=X.csv", skiprows=3, encoding="utf-8")

# カラム名の設定
cny_jpy_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
php_jpy_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
vnd_jpy_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# 日付の処理
cny_jpy_df["Date"] = pd.to_datetime(cny_jpy_df["Date"])
php_jpy_df["Date"] = pd.to_datetime(php_jpy_df["Date"])
vnd_jpy_df["Date"] = pd.to_datetime(vnd_jpy_df["Date"])

# 週情報から日を特定する関数
def get_day_from_week(year, month, week):
    if year < 18 and month < 7:
        # 古いデータ：上旬中旬下旬の週番号に対応
        if week == 1:
            return 5
        elif week == 2:
            return 15
        else:
            return 25
    
    # 新しいデータ：週番号（1週目の月曜日 + 7日 * (week - 1)）
    if pd.isna(week):
        return 1  # 週情報がない場合はデフォルトで1日
    else:
        return min(((int(week) - 1) * 7) + 1, 28)  # 月末を超えないよう28日を上限に

# カプセルトイデータに日付情報を追加
capsule_df["day"] = capsule_df.apply(
    lambda row: get_day_from_week(row["year"], row["month"], row["week"]), axis=1
)

capsule_df["Date"] = capsule_df.apply(
    lambda row: pd.to_datetime(
        f"{int(row['year']) + 2000}-{str(int(row['month'])).zfill(2)}-{str(int(row['day'])).zfill(2)}"
    ),
    axis=1,
)

# カプセルトイ価格を日付ごとに平均（元の月ごとではなく日ごとに）
daily_capsule = capsule_df.groupby("Date").agg({"price": "mean"}).reset_index()

# 為替データを結合して日付ベースのデータフレームを作成
cny_jpy_daily = cny_jpy_df[["Date", "Close"]].rename(columns={"Close": "CNY_JPY"})
php_jpy_daily = php_jpy_df[["Date", "Close"]].rename(columns={"Close": "PHP_JPY"})
vnd_jpy_daily = vnd_jpy_df[["Date", "Close"]].rename(columns={"Close": "VND_JPY"})

# 為替データを統合
exchange_daily = pd.merge(cny_jpy_daily, php_jpy_daily, on="Date", how="outer")
exchange_daily = pd.merge(exchange_daily, vnd_jpy_daily, on="Date", how="outer")

# 欠損値を前方補間 (forward fill) と後方補間 (backward fill) の組み合わせで埋める
exchange_daily = exchange_daily.sort_values("Date")
exchange_daily = exchange_daily.ffill().bfill()

# カプセルトイデータの日付に最も近い為替データを割り当てる関数
def assign_nearest_exchange_rate(capsule_date, exchange_df):
    # 引数の日付に最も近い為替データの日付を検索
    nearest_date = exchange_df["Date"].iloc[(exchange_df["Date"] - capsule_date).abs().argsort()[0]]
    return exchange_df[exchange_df["Date"] == nearest_date].iloc[0]

# カプセルトイデータに最も近い為替データを割り当て
result_data = []
for _, capsule_row in daily_capsule.iterrows():
    capsule_date = capsule_row["Date"]
    
    # 同じ日付の為替データがあるか確認
    matching_exchange = exchange_daily[exchange_daily["Date"] == capsule_date]
    
    if not matching_exchange.empty:
        # 同じ日付のデータがある場合
        exchange_data = matching_exchange.iloc[0]
    else:
        # 同じ日付のデータがない場合、最も近い日付のデータを使用
        exchange_data = assign_nearest_exchange_rate(capsule_date, exchange_daily)
    
    result_data.append({
        "Date": capsule_date,
        "price": capsule_row["price"],
        "CNY_JPY": exchange_data["CNY_JPY"],
        "PHP_JPY": exchange_data["PHP_JPY"],
        "VND_JPY": exchange_data["VND_JPY"]
    })

# 結果のデータフレームを作成
df = pd.DataFrame(result_data)

# 元データの基本統計量の確認
print("元データの基本統計量:")
print(df.describe())

# 元データのスケール違いを可視化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df[["CNY_JPY", "PHP_JPY", "VND_JPY"]])
plt.title("元の為替レートのスケール比較")
plt.ylabel("為替レート")
plt.grid(True)

plt.subplot(1, 2, 2)
sns.boxplot(data=df[["price"]])
plt.title("カプセルトイ価格のスケール")
plt.grid(True)
plt.tight_layout()
plt.savefig("original_scales.png")
plt.close()

# データを変化率に変換
df_pct_change = df.copy().sort_values("Date")

# 元の値を別の列に保存
change_cols = ["price", "CNY_JPY", "PHP_JPY", "VND_JPY"]
for col in change_cols:
    df_pct_change[f"{col}_original"] = df_pct_change[col]

# 変化率を計算（前日比）
df_pct_change[change_cols] = df_pct_change[change_cols].pct_change() * 100  # パーセンテージで表示

# 最初の行（NaN）を削除
df_pct_change = df_pct_change.dropna()

# 変化率データの基本統計量
print("\n変化率データの基本統計量:")
print(df_pct_change[change_cols].describe())

# 変化率データのスケール比較を可視化
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_pct_change[change_cols])
plt.title("変化率データのスケール比較")
plt.ylabel("変化率 (%)")
plt.grid(True)
plt.savefig("pct_change_scales.png")
plt.close()

# 変化率の相関係数
correlation_pct = df_pct_change[change_cols].corr()
print("\n変化率の相関係数:")
print(correlation_pct)

# 変化率の相関行列ヒートマップ
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_pct, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("変化率データの相関係数ヒートマップ")
plt.savefig("pct_change_correlation.png")
plt.close()

# 時系列での変化率の推移
plt.figure(figsize=(14, 10))

# カプセルトイ価格の変化率
plt.subplot(2, 1, 1)
plt.plot(df_pct_change["Date"], df_pct_change["price"], marker='o', linestyle='-', color='blue')
plt.title("カプセルトイ価格の変化率 (%)")
plt.ylabel("価格変化率 (%)")
plt.grid(True)
plt.xticks(rotation=45)

# 為替レートの変化率
plt.subplot(2, 1, 2)
plt.plot(df_pct_change["Date"], df_pct_change["CNY_JPY"], marker='o', linestyle='-', label='CNY/JPY')
plt.plot(df_pct_change["Date"], df_pct_change["PHP_JPY"], marker='x', linestyle='--', label='PHP/JPY')
plt.plot(df_pct_change["Date"], df_pct_change["VND_JPY"], marker='^', linestyle='-.', label='VND/JPY')
plt.title("為替レートの変化率 (%)")
plt.ylabel("為替変化率 (%)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig("pct_change_timeseries.png")
plt.close()

# 変化率データの散布図行列
plt.figure(figsize=(14, 10))
sns.pairplot(df_pct_change[change_cols])
plt.suptitle('変化率データの散布図行列', y=1.02)
plt.savefig("pct_change_pairplot.png")
plt.close()

# 変化率での回帰分析
X_pct = df_pct_change[["CNY_JPY", "PHP_JPY", "VND_JPY"]]
y_pct = df_pct_change["price"]

# トレーニングデータとテストデータに分割
X_train_pct, X_test_pct, y_train_pct, y_test_pct = train_test_split(
    X_pct, y_pct, test_size=0.2, random_state=42
)

# モデルの構築
model_pct = LinearRegression()
model_pct.fit(X_train_pct, y_train_pct)

# 係数と切片
print(f"\n変化率モデルの切片: {model_pct.intercept_:.4f}")
for i, col in enumerate(X_pct.columns):
    print(f"{col}の係数: {model_pct.coef_[i]:.4f}")

# 定数項を追加
X_train_sm_pct = sm.add_constant(X_train_pct)

# モデルの構築
model_sm_pct = sm.OLS(y_train_pct, X_train_sm_pct).fit()

# 詳細な統計結果
print("\n変化率モデルの詳細:")
print(model_sm_pct.summary())

# 予測の実行
X_test_sm_pct = sm.add_constant(X_test_pct)
y_pred_pct = model_sm_pct.predict(X_test_sm_pct)

# 評価指標
mse_pct = mean_squared_error(y_test_pct, y_pred_pct)
r2_pct = r2_score(y_test_pct, y_pred_pct)
print(f"\n変化率モデルの平均二乗誤差: {mse_pct:.4f}")
print(f"変化率モデルの決定係数 (R²): {r2_pct:.4f}")

# 残差
residuals_pct = y_test_pct - y_pred_pct

# 残差プロット
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_pct, residuals_pct)
plt.axhline(y=0, color="r", linestyle="-")
plt.title("変化率モデルの残差プロット")
plt.xlabel("予測値")
plt.ylabel("残差")
plt.grid(True)
plt.savefig("pct_change_residuals.png")
plt.close()

# QQプロット（正規性の確認）
fig = plt.figure(figsize=(10, 6))
sm.qqplot(residuals_pct, line="45", fit=True, ax=plt.gca())
plt.title("変化率モデル残差のQQプロット")
plt.grid(True)
plt.savefig("pct_change_qqplot.png")
plt.close()

# VIF (分散拡大要因)の計算
X_vif_pct = sm.add_constant(X_pct)
vif_data_pct = pd.DataFrame()
vif_data_pct["変数"] = X_vif_pct.columns
vif_data_pct["VIF"] = [variance_inflation_factor(X_vif_pct.values, i) for i in range(X_vif_pct.shape[1])]
print("\n変化率モデルのVIF (分散拡大要因):")
print(vif_data_pct)

# 全データセットでの予測
X_all_pct = sm.add_constant(X_pct)
df_pct_change['Predicted'] = model_sm_pct.predict(X_all_pct)

# 実際の変化率と予測変化率の比較
plt.figure(figsize=(12, 6))
plt.plot(df_pct_change["Date"], df_pct_change['price'], label='実際の価格変化率 (%)', marker='o')
plt.plot(df_pct_change["Date"], df_pct_change['Predicted'], label='予測価格変化率 (%)', linestyle='--', marker='x')
plt.title('カプセルトイ価格変化率: 実際 vs 予測')
plt.xlabel('日付')
plt.ylabel('価格変化率 (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("pct_change_actual_vs_predicted.png")
plt.close()

# 各為替レート変化率の影響を可視化
coef_df_pct = pd.DataFrame({
    '変数': X_pct.columns,
    '係数': model_pct.coef_
})
plt.figure(figsize=(10, 6))
bars = sns.barplot(x='変数', y='係数', data=coef_df_pct)

# 係数の値を棒グラフの上に表示
for i, p in enumerate(bars.patches):
    bars.annotate(f'{coef_df_pct["係数"].iloc[i]:.4f}', 
                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha = 'center', va = 'bottom', 
                  xytext = (0, 5), textcoords = 'offset points')

plt.title('各為替レート変化率がカプセルトイ価格変化率に与える影響度')
plt.xlabel('為替レート')
plt.ylabel('係数の大きさ')
plt.grid(True, axis='y')
plt.savefig("pct_change_coefficients.png")
plt.close()

# 月ごとの集計分析 (変化率の平均を計算)
df_pct_change['Year'] = df_pct_change['Date'].dt.year
df_pct_change['Month'] = df_pct_change['Date'].dt.month
monthly_avg = df_pct_change.groupby(['Year', 'Month'])[change_cols].mean().reset_index()

# 月ごとの変化率の相関
monthly_corr = monthly_avg[change_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(monthly_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("月次平均変化率の相関係数ヒートマップ")
plt.savefig("monthly_pct_change_correlation.png")
plt.close()

# 月ごとの変化率の時系列プロット
monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))

plt.figure(figsize=(14, 10))

# カプセルトイ価格の月次平均変化率
plt.subplot(2, 1, 1)
plt.plot(monthly_avg["Date"], monthly_avg["price"], marker='o', linestyle='-', color='blue')
plt.title("カプセルトイ価格の月次平均変化率 (%)")
plt.ylabel("価格変化率 (%)")
plt.grid(True)
plt.xticks(rotation=45)

# 為替レートの月次平均変化率
plt.subplot(2, 1, 2)
plt.plot(monthly_avg["Date"], monthly_avg["CNY_JPY"], marker='o', linestyle='-', label='CNY/JPY')
plt.plot(monthly_avg["Date"], monthly_avg["PHP_JPY"], marker='x', linestyle='--', label='PHP/JPY')
plt.plot(monthly_avg["Date"], monthly_avg["VND_JPY"], marker='^', linestyle='-.', label='VND/JPY')
plt.title("為替レートの月次平均変化率 (%)")
plt.ylabel("為替変化率 (%)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig("monthly_pct_change_timeseries.png")
plt.close()

# 結果のCSV出力
df_pct_change.to_csv("exchange_rate_pct_change_analysis.csv", index=False)
monthly_avg.to_csv("monthly_exchange_rate_pct_change.csv", index=False)

# 相関分析の結果の要約をテキストファイルに保存
with open("pct_change_analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("# 変化率分析の結果要約\n\n")
    
    f.write("## 1. 基本統計量\n")
    f.write(df_pct_change[change_cols].describe().to_string())
    f.write("\n\n")
    
    f.write("## 2. 変化率の相関係数\n")
    f.write(correlation_pct.to_string())
    f.write("\n\n")
    
    f.write("## 3. 回帰モデルの詳細\n")
    f.write(f"切片: {model_pct.intercept_:.4f}\n")
    for i, col in enumerate(X_pct.columns):
        f.write(f"{col}の係数: {model_pct.coef_[i]:.4f}\n")
    f.write("\n")
    
    f.write("## 4. モデル評価指標\n")
    f.write(f"平均二乗誤差 (MSE): {mse_pct:.4f}\n")
    f.write(f"決定係数 (R²): {r2_pct:.4f}\n")
    f.write("\n")
    
    f.write("## 5. 多重共線性の評価 (VIF)\n")
    f.write(vif_data_pct.to_string())
    f.write("\n\n")
    
    f.write("## 6. 回帰モデルの詳細な統計結果\n")
    f.write(model_sm_pct.summary().as_text())

print("分析完了！ファイルが保存されました。")