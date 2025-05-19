import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# 為替データの変化率(%)を計算
exchange_daily_pct = exchange_daily.copy()
for col in ["CNY_JPY", "PHP_JPY", "VND_JPY"]:
    exchange_daily_pct[f"{col}_pct_change"] = exchange_daily[col].pct_change() * 100
    # 最初の行のNaNを0に置き換え
    exchange_daily_pct[f"{col}_pct_change"].fillna(0, inplace=True)

# スケーラーを定義
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

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
    matching_exchange_pct = exchange_daily_pct[exchange_daily_pct["Date"] == capsule_date]
    
    if not matching_exchange.empty:
        # 同じ日付のデータがある場合
        exchange_data = matching_exchange.iloc[0]
        exchange_pct_data = matching_exchange_pct.iloc[0]
    else:
        # 同じ日付のデータがない場合、最も近い日付のデータを使用
        exchange_data = assign_nearest_exchange_rate(capsule_date, exchange_daily)
        exchange_pct_data = assign_nearest_exchange_rate(capsule_date, exchange_daily_pct)
    
    result_data.append({
        "Date": capsule_date,
        "price": capsule_row["price"],
        "CNY_JPY": exchange_data["CNY_JPY"],
        "PHP_JPY": exchange_data["PHP_JPY"],
        "VND_JPY": exchange_data["VND_JPY"],
        "CNY_JPY_pct_change": exchange_pct_data["CNY_JPY_pct_change"] if "CNY_JPY_pct_change" in exchange_pct_data else 0,
        "PHP_JPY_pct_change": exchange_pct_data["PHP_JPY_pct_change"] if "PHP_JPY_pct_change" in exchange_pct_data else 0,
        "VND_JPY_pct_change": exchange_pct_data["VND_JPY_pct_change"] if "VND_JPY_pct_change" in exchange_pct_data else 0
    })

# 結果のデータフレームを作成
df = pd.DataFrame(result_data)

# 標準化したデータフレームを作成
df_standardized = df.copy()
# 価格と為替レートを標準化
exchange_cols = ["CNY_JPY", "PHP_JPY", "VND_JPY"]
scaled_data = standard_scaler.fit_transform(df[exchange_cols])
df_standardized[exchange_cols] = scaled_data

# Min-Max スケーリングしたデータフレームを作成
df_minmax = df.copy()
# 価格と為替レートをMin-Maxスケーリング（0-1の範囲に）
minmax_data = minmax_scaler.fit_transform(df[exchange_cols])
df_minmax[exchange_cols] = minmax_data

# 変化率データフレーム
df_pct_change = df.copy()


# 基本統計量の確認
print("基本統計量:")
print(df.describe())

# 1. オリジナルデータでの相関分析
correlation_original = df[["price", "CNY_JPY", "PHP_JPY", "VND_JPY"]].corr()
print("\n1. オリジナルデータの相関係数:")
print(correlation_original)

# 2. 標準化データでの相関分析
correlation_standardized = df_standardized[["price", "CNY_JPY", "PHP_JPY", "VND_JPY"]].corr()
print("\n2. 標準化データの相関係数:")
print(correlation_standardized)

# 3. Min-Maxスケーリングデータでの相関分析
correlation_minmax = df_minmax[["price", "CNY_JPY", "PHP_JPY", "VND_JPY"]].corr()
print("\n3. Min-Maxスケーリングデータの相関係数:")
print(correlation_minmax)

# 4. 変化率データでの相関分析
correlation_pct_change = df[["price", "CNY_JPY_pct_change", "PHP_JPY_pct_change", "VND_JPY_pct_change"]].corr()
print("\n4. 変化率データの相関係数:")
print(correlation_pct_change)

# ヒートマップを表示する関数
def plot_heatmap(correlation, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 各種相関ヒートマップを表示
plot_heatmap(correlation_original, "1. オリジナルデータの相関係数ヒートマップ")
plot_heatmap(correlation_standardized, "2. 標準化データの相関係数ヒートマップ")
plot_heatmap(correlation_minmax, "3. Min-Maxスケーリングデータの相関係数ヒートマップ")
plot_heatmap(correlation_pct_change, "4. 変化率データの相関係数ヒートマップ")

# 時系列トレンドの確認
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["price"])
plt.title("カプセルトイ価格の日次時系列推移")
plt.xlabel("日付")
plt.ylabel("価格 (円)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 各手法での回帰分析を行う関数
def run_regression_analysis(X, y, title):
    print(f"\n--- {title} ---")
    
    # トレーニングデータとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # モデルの構築
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 係数と切片
    print(f"切片: {model.intercept_:.4f}")
    for i, col in enumerate(X.columns):
        print(f"{col}の係数: {model.coef_[i]:.4f}")
    
    # 定数項を追加
    X_train_sm = sm.add_constant(X_train)
    
    # モデルの構築
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    
    # 詳細な統計結果
    print("\n回帰モデルの詳細:")
    print(model_sm.summary())
    
    # 予測の実行
    X_test_sm = sm.add_constant(X_test)
    y_pred = model_sm.predict(X_test_sm)
    
    # 評価指標
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n平均二乗誤差: {mse:.4f}")
    print(f"決定係数 (R²): {r2:.4f}")
    
    # VIF (分散拡大要因)の計算
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X_vif = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["変数"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print("\nVIF (分散拡大要因):")
    print(vif_data)
    
    # 係数の解釈
    coef_df = pd.DataFrame({
        '変数': X.columns,
        '係数': model.coef_
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='変数', y='係数', data=coef_df)
    plt.title(f'各変数がカプセルトイ価格に与える影響度 ({title})')
    plt.xlabel('変数')
    plt.ylabel('係数の大きさ')
    plt.grid(True, axis='y')
    plt.show()
    
    return model, r2

# 1. オリジナルデータでの回帰分析
X_original = df[["CNY_JPY", "PHP_JPY", "VND_JPY"]]
y = df["price"]
model_original, r2_original = run_regression_analysis(X_original, y, "オリジナルデータ")

# 2. 標準化データでの回帰分析
X_standardized = df_standardized[["CNY_JPY", "PHP_JPY", "VND_JPY"]]
model_standardized, r2_standardized = run_regression_analysis(X_standardized, y, "標準化データ")

# 3. 変化率データでの回帰分析
X_pct_change = df[["CNY_JPY_pct_change", "PHP_JPY_pct_change", "VND_JPY_pct_change"]]
model_pct_change, r2_pct_change = run_regression_analysis(X_pct_change, y, "変化率データ")

# 日次データと為替データの散布図行列
plt.figure(figsize=(14, 10))
sns.pairplot(df[["price", "CNY_JPY", "PHP_JPY", "VND_JPY"]])
plt.suptitle('カプセルトイ価格と為替レートの散布図行列', y=1.02)
plt.show()

# 時系列での為替レートとカプセルトイ価格の推移（複数軸）
fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:blue'
ax1.set_xlabel('日付')
ax1.set_ylabel('カプセルトイ価格 (円)', color=color)
ax1.plot(df["Date"], df['price'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# 2つ目のY軸を作成
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('為替レート', color=color)
ax2.plot(df["Date"], df['CNY_JPY'], color='tab:red', label='CNY/JPY')
ax2.plot(df["Date"], df['PHP_JPY'], color='tab:green', label='PHP/JPY')
ax2.plot(df["Date"], df['VND_JPY'], color='tab:orange', label='VND/JPY')
ax2.tick_params(axis='y', labelcolor=color)

# 凡例を追加
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, ['カプセルトイ価格'] + labels2, loc='upper left')

plt.title('カプセルトイ価格と為替レートの時系列推移')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()