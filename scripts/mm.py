import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

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

# ===== MinMaxスケーリングを適用 =====
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

# MinMaxスケーリングを適用するデータフレームを作成
df_minmax = df.copy()

# スケーリングする列
cols_to_scale = ["price", "CNY_JPY", "PHP_JPY", "VND_JPY"]

# 元のデータを保存
for col in cols_to_scale:
    df_minmax[f"{col}_original"] = df_minmax[col]

# MinMaxスケーラーを初期化
scaler = MinMaxScaler()

# スケーリングを適用
df_minmax[cols_to_scale] = scaler.fit_transform(df_minmax[cols_to_scale])

# MinMaxスケーリング後のデータの基本統計量
print("\nMinMaxスケーリング後のデータの基本統計量:")
print(df_minmax[cols_to_scale].describe())

# スケーリング後のデータのスケール比較を可視化
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_minmax[cols_to_scale])
plt.title("MinMaxスケーリング後のデータのスケール比較")
plt.ylabel("スケーリング値 (0-1)")
plt.grid(True)
plt.savefig("minmax_scales.png")
plt.close()

# MinMaxスケーリング後の相関係数
correlation_minmax = df_minmax[cols_to_scale].corr()
print("\nMinMaxスケーリング後の相関係数:")
print(correlation_minmax)

# 相関行列ヒートマップ
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_minmax, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("MinMaxスケーリング後のデータの相関係数ヒートマップ")
plt.savefig("minmax_correlation.png")
plt.close()

# 時系列でのスケーリング後のデータの推移
plt.figure(figsize=(14, 10))

# カプセルトイ価格（スケーリング後）
plt.subplot(2, 1, 1)
plt.plot(df_minmax["Date"], df_minmax["price"], marker='o', linestyle='-', color='blue')
plt.title("カプセルトイ価格（MinMaxスケーリング後）")
plt.ylabel("スケーリング値 (0-1)")
plt.grid(True)
plt.xticks(rotation=45)

# 為替レート（スケーリング後）
plt.subplot(2, 1, 2)
plt.plot(df_minmax["Date"], df_minmax["CNY_JPY"], marker='o', linestyle='-', label='CNY/JPY')
plt.plot(df_minmax["Date"], df_minmax["PHP_JPY"], marker='x', linestyle='--', label='PHP/JPY')
plt.plot(df_minmax["Date"], df_minmax["VND_JPY"], marker='^', linestyle='-.', label='VND/JPY')
plt.title("為替レート（MinMaxスケーリング後）")
plt.ylabel("スケーリング値 (0-1)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig("minmax_timeseries.png")
plt.close()

# スケーリング後のデータの散布図行列
plt.figure(figsize=(14, 10))
sns.pairplot(df_minmax[cols_to_scale])
plt.suptitle('MinMaxスケーリング後のデータの散布図行列', y=1.02)
plt.savefig("minmax_pairplot.png")
plt.close()

# スケーリング後のデータでの回帰分析
X_minmax = df_minmax[["CNY_JPY", "PHP_JPY", "VND_JPY"]]
y_minmax = df_minmax["price"]

# トレーニングデータとテストデータに分割
X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(
    X_minmax, y_minmax, test_size=0.2, random_state=42
)

# モデルの構築
model_minmax = LinearRegression()
model_minmax.fit(X_train_minmax, y_train_minmax)

# 係数と切片
print(f"\nMinMaxスケーリングモデルの切片: {model_minmax.intercept_:.4f}")
for i, col in enumerate(X_minmax.columns):
    print(f"{col}の係数: {model_minmax.coef_[i]:.4f}")

# 定数項を追加
X_train_sm_minmax = sm.add_constant(X_train_minmax)

# モデルの構築
model_sm_minmax = sm.OLS(y_train_minmax, X_train_sm_minmax).fit()

# 詳細な統計結果
print("\nMinMaxスケーリングモデルの詳細:")
print(model_sm_minmax.summary())

# 予測の実行
X_test_sm_minmax = sm.add_constant(X_test_minmax)
y_pred_minmax = model_sm_minmax.predict(X_test_sm_minmax)

# 評価指標
mse_minmax = mean_squared_error(y_test_minmax, y_pred_minmax)
r2_minmax = r2_score(y_test_minmax, y_pred_minmax)
print(f"\nMinMaxスケーリングモデルの平均二乗誤差: {mse_minmax:.4f}")
print(f"MinMaxスケーリングモデルの決定係数 (R²): {r2_minmax:.4f}")

# 残差
residuals_minmax = y_test_minmax - y_pred_minmax

# 残差プロット
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_minmax, residuals_minmax)
plt.axhline(y=0, color="r", linestyle="-")
plt.title("MinMaxスケーリングモデルの残差プロット")
plt.xlabel("予測値")
plt.ylabel("残差")
plt.grid(True)
plt.savefig("minmax_residuals.png")
plt.close()

# QQプロット（正規性の確認）
fig = plt.figure(figsize=(10, 6))
sm.qqplot(residuals_minmax, line="45", fit=True, ax=plt.gca())
plt.title("MinMaxスケーリングモデル残差のQQプロット")
plt.grid(True)
plt.savefig("minmax_qqplot.png")
plt.close()

# VIF (分散拡大要因)の計算
X_vif_minmax = sm.add_constant(X_minmax)
vif_data_minmax = pd.DataFrame()
vif_data_minmax["変数"] = X_vif_minmax.columns
vif_data_minmax["VIF"] = [variance_inflation_factor(X_vif_minmax.values, i) for i in range(X_vif_minmax.shape[1])]
print("\nMinMaxスケーリングモデルのVIF (分散拡大要因):")
print(vif_data_minmax)

# 全データセットでの予測
X_all_minmax = sm.add_constant(X_minmax)
df_minmax['Predicted'] = model_sm_minmax.predict(X_all_minmax)

# 実際の値と予測値の比較（スケーリング後）
plt.figure(figsize=(12, 6))
plt.plot(df_minmax["Date"], df_minmax['price'], label='実際の価格（スケーリング後）', marker='o')
plt.plot(df_minmax["Date"], df_minmax['Predicted'], label='予測価格（スケーリング後）', linestyle='--', marker='x')
plt.title('カプセルトイ価格: 実際 vs 予測（MinMaxスケーリング後）')
plt.xlabel('日付')
plt.ylabel('スケーリング値 (0-1)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("minmax_actual_vs_predicted.png")
plt.close()

# 各為替レート（スケーリング後）の影響を可視化
coef_df_minmax = pd.DataFrame({
    '変数': X_minmax.columns,
    '係数': model_minmax.coef_
})
plt.figure(figsize=(10, 6))
bars = sns.barplot(x='変数', y='係数', data=coef_df_minmax)

# 係数の値を棒グラフの上に表示
for i, p in enumerate(bars.patches):
    bars.annotate(f'{coef_df_minmax["係数"].iloc[i]:.4f}', 
                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha = 'center', va = 'bottom', 
                  xytext = (0, 5), textcoords = 'offset points')

plt.title('各為替レートがカプセルトイ価格に与える影響度（MinMaxスケーリング後）')
plt.xlabel('為替レート')
plt.ylabel('係数の大きさ')
plt.grid(True, axis='y')
plt.savefig("minmax_coefficients.png")
plt.close()

# 月ごとの集計分析（スケーリング後のデータ）
df_minmax['Year'] = df_minmax['Date'].dt.year
df_minmax['Month'] = df_minmax['Date'].dt.month
monthly_avg_minmax = df_minmax.groupby(['Year', 'Month'])[cols_to_scale].mean().reset_index()

# 月ごとのスケーリングデータの相関
monthly_corr_minmax = monthly_avg_minmax[cols_to_scale].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(monthly_corr_minmax, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("月次データの相関係数ヒートマップ（MinMaxスケーリング後）")
plt.savefig("monthly_minmax_correlation.png")
plt.close()

# 月ごとのスケーリングデータの時系列プロット
monthly_avg_minmax['Date'] = pd.to_datetime(monthly_avg_minmax[['Year', 'Month']].assign(day=1))

plt.figure(figsize=(14, 10))

# カプセルトイ価格の月次平均（スケーリング後）
plt.subplot(2, 1, 1)
plt.plot(monthly_avg_minmax["Date"], monthly_avg_minmax["price"], marker='o', linestyle='-', color='blue')
plt.title("カプセルトイ価格の月次平均（MinMaxスケーリング後）")
plt.ylabel("スケーリング値 (0-1)")
plt.grid(True)
plt.xticks(rotation=45)

# 為替レートの月次平均（スケーリング後）
plt.subplot(2, 1, 2)
plt.plot(monthly_avg_minmax["Date"], monthly_avg_minmax["CNY_JPY"], marker='o', linestyle='-', label='CNY/JPY')
plt.plot(monthly_avg_minmax["Date"], monthly_avg_minmax["PHP_JPY"], marker='x', linestyle='--', label='PHP/JPY')
plt.plot(monthly_avg_minmax["Date"], monthly_avg_minmax["VND_JPY"], marker='^', linestyle='-.', label='VND/JPY')
plt.title("為替レートの月次平均（MinMaxスケーリング後）")
plt.ylabel("スケーリング値 (0-1)")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig("monthly_minmax_timeseries.png")
plt.close()

# 元のスケールと比較するための散布図（原価格 vs 為替レート）
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(df_minmax["CNY_JPY_original"], df_minmax["price_original"], alpha=0.7)
plt.title("カプセルトイ価格 vs CNY/JPY（元のスケール）")
plt.xlabel("CNY/JPY")
plt.ylabel("カプセルトイ価格 (円)")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(df_minmax["PHP_JPY_original"], df_minmax["price_original"], alpha=0.7)
plt.title("カプセルトイ価格 vs PHP/JPY（元のスケール）")
plt.xlabel("PHP/JPY")
plt.ylabel("カプセルトイ価格 (円)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(df_minmax["VND_JPY_original"], df_minmax["price_original"], alpha=0.7)
plt.title("カプセルトイ価格 vs VND/JPY（元のスケール）")
plt.xlabel("VND/JPY")
plt.ylabel("カプセルトイ価格 (円)")
plt.grid(True)

plt.tight_layout()
plt.savefig("original_scale_scatter.png")
plt.close()

# スケーリング後の散布図（スケーリング後の価格 vs スケーリング後の為替レート）
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(df_minmax["CNY_JPY"], df_minmax["price"], alpha=0.7)
plt.title("カプセルトイ価格 vs CNY/JPY（MinMaxスケーリング後）")
plt.xlabel("CNY/JPY（スケーリング値）")
plt.ylabel("カプセルトイ価格（スケーリング値）")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(df_minmax["PHP_JPY"], df_minmax["price"], alpha=0.7)
plt.title("カプセルトイ価格 vs PHP/JPY（MinMaxスケーリング後）")
plt.xlabel("PHP/JPY（スケーリング値）")
plt.ylabel("カプセルトイ価格（スケーリング値）")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(df_minmax["VND_JPY"], df_minmax["price"], alpha=0.7)
plt.title("カプセルトイ価格 vs VND/JPY（MinMaxスケーリング後）")
plt.xlabel("VND/JPY（スケーリング値）")
plt.ylabel("カプセルトイ価格（スケーリング値）")
plt.grid(True)

plt.tight_layout()
plt.savefig("minmax_scale_scatter.png")
plt.close()

# 結果のCSV出力
df_minmax.to_csv("exchange_rate_minmax_scaling_analysis.csv", index=False)
monthly_avg_minmax.to_csv("monthly_exchange_rate_minmax_scaling.csv", index=False)

# 分析結果の要約をテキストファイルに保存
with open("minmax_scaling_analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("# MinMaxスケーリング分析の結果要約\n\n")
    
    f.write("## 1. 基本統計量（スケーリング後）\n")
    f.write(df_minmax[cols_to_scale].describe().to_string())
    f.write("\n\n")
    
    f.write("## 2. 相関係数（スケーリング後）\n")
    f.write(correlation_minmax.to_string())
    f.write("\n\n")
    
    f.write("## 3. 回帰モデルの詳細\n")
    f.write(f"切片: {model_minmax.intercept_:.4f}\n")
    for i, col in enumerate(X_minmax.columns):
        f.write(f"{col}の係数: {model_minmax.coef_[i]:.4f}\n")
    f.write("\n")
    
    f.write("## 4. モデル評価指標\n")
    f.write(f"平均二乗誤差 (MSE): {mse_minmax:.4f}\n")
    f.write(f"決定係数 (R²): {r2_minmax:.4f}\n")
    f.write("\n")
    
    f.write("## 5. 多重共線性の評価 (VIF)\n")
    f.write(vif_data_minmax.to_string())
    f.write("\n\n")
    
    f.write("## 6. 回帰モデルの詳細な統計結果\n")
    f.write(model_sm_minmax.summary().as_text())
    
    f.write("\n\n## 7. MinMaxスケーリングの特徴\n")
    f.write("- MinMaxスケーリングはすべての変数を0〜1の範囲に変換します\n")
    f.write("- スケール変換前後の相関係数の値は同じですが、回帰分析の結果には影響があります\n")
    f.write("- 外れ値の影響を受けやすいため、外れ値がある場合は注意が必要です\n")
    
    f.write("\n\n## 8. 分析の結論\n")
    f.write("- 為替レートとカプセルトイ価格の関係を同じスケールで比較することができました\n")
    f.write("- 係数の絶対値を比較することで、各為替レートの相対的な影響度を評価できます\n")
    if r2_minmax < 0.3:
        f.write("- モデルの決定係数が低いため、為替レート以外の要因も価格に大きく影響している可能性があります\n")
    elif r2_minmax >= 0.7:
        f.write("- モデルの決定係数が高く、為替レートがカプセルトイ価格に強く影響していることが示唆されます\n")

print("MinMaxスケーリング分析が完了しました！ファイルが保存されました。")