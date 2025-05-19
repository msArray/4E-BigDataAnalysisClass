import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

# 日本語フォントの設定
plt.rcParams["font.family"] = "MS Gothic"

# データの読み込み
df = pd.read_csv("./merged_all.csv", encoding="utf-8")
df.columns = ["Date", "Name", "Price", "Category", "CNY", "PHP", "VND", "Gasoline"]

# データの確認
print("データの概要:")
print(df.head())
print(f"\nデータの行数: {len(df)}")
print("\nデータの基本統計量:")
print(df.describe())

# Dateカラムを日付型に変換
df["Date"] = pd.to_datetime(df["Date"])

# 外れ値の確認
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df["Price"])
plt.title("価格のボックスプロット")
plt.ylabel("価格")

plt.subplot(1, 2, 2)
sns.histplot(df["Price"], kde=True)
plt.title("価格のヒストグラム")
plt.xlabel("価格")
plt.tight_layout()
plt.show()

# 外れ値処理（IQR法）
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n外れ値の閾値 - 下限: {lower_bound}, 上限: {upper_bound}")
outliers = df[(df["Price"] < lower_bound) | (df["Price"] > upper_bound)]
print(f"外れ値の数: {len(outliers)}, 全体の {len(outliers)/len(df)*100:.2f}%")

# 外れ値をクリッピング
df_cleaned = df.copy()
df_cleaned["Price"] = df_cleaned["Price"].clip(lower=lower_bound, upper=upper_bound)

# 特徴量間の相関を確認
numeric_cols = ["Price", "CNY", "PHP", "VND", "Gasoline"]
correlation_matrix = df_cleaned[numeric_cols].corr()
print("\n相関行列:")
print(correlation_matrix)

# 相関行列のヒートマップ
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, annot=True, cmap=sns.color_palette("Reds", 10), fmt=".2f"
)
plt.title("相関行列ヒートマップ")
plt.tight_layout()
plt.show()

# 数値型の特徴量を選択
X = df_cleaned[["CNY", "PHP", "VND", "Gasoline"]]
y = df_cleaned["Price"]


# 多重共線性のチェック
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data


print("\n多重共線性のチェック (VIF):")
print(calculate_vif(X))
print("VIF > 10は多重共線性が強いことを示します")

# スケーリング処理
# StandardScalerとRobustScalerを両方試す
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

X_standard = standard_scaler.fit_transform(X)
X_robust = robust_scaler.fit_transform(X)

# スケーリング結果の確認
X_standard_df = pd.DataFrame(X_standard, columns=X.columns)
X_robust_df = pd.DataFrame(X_robust, columns=X.columns)

print("\nStandardScalerでのスケーリング後のデータ統計:")
print(X_standard_df.describe())

print("\nRobustScalerでのスケーリング後のデータ統計:")
print(X_robust_df.describe())

# 訓練データとテストデータに分割（サンプル数が多いので通常の分割）
X_train_std, X_test_std, y_train, y_test = train_test_split(
    X_standard, y, test_size=0.2, random_state=42
)

X_train_rob, X_test_rob, _, _ = train_test_split(
    X_robust, y, test_size=0.2, random_state=42
)

# 異なるモデルを比較
models = {
    "LinearRegression (標準化)": LinearRegression(),
    "Ridge (標準化)": Ridge(alpha=1.0),
    "Lasso (標準化)": Lasso(alpha=0.1),
    "LinearRegression (頑健化)": LinearRegression(),
    "Ridge (頑健化)": Ridge(alpha=1.0),
    "Lasso (頑健化)": Lasso(alpha=0.1),
}

results = {}

# 各モデルをトレーニングと評価
for i, (name, model) in enumerate(models.items()):
    if "標準化" in name:
        X_train, X_test = X_train_std, X_test_std
    else:
        X_train, X_test = X_train_rob, X_test_rob

    # モデルの訓練
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 結果を保存
    results[name] = {"model": model, "mse": mse, "r2": r2, "y_pred": y_pred}

    print(f"\n{name}の結果:")
    print(f"平均二乗誤差 (MSE): {mse:.4f}")
    print(f"決定係数 (R²): {r2:.4f}")

    # 係数（スケーリング前の単位に戻す）
    print(f"\n{name}の回帰係数:")
    if "標準化" in name:
        coeffs = model.coef_ / standard_scaler.scale_
        for feature, coef in zip(X.columns, coeffs):
            print(f"{feature}: {coef:.4f}")
        intercept = model.intercept_ - np.sum(
            model.coef_ * standard_scaler.mean_ / standard_scaler.scale_
        )
        print(f"切片: {intercept:.4f}")
    else:
        coeffs = model.coef_ / robust_scaler.scale_
        for feature, coef in zip(X.columns, coeffs):
            print(f"{feature}: {coef:.4f}")
        intercept = model.intercept_ - np.sum(
            model.coef_ * robust_scaler.center_ / robust_scaler.scale_
        )
        print(f"切片: {intercept:.4f}")

# 最良モデルを選択
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]
best_y_pred = results[best_model_name]["y_pred"]

print(f"\n最良モデル: {best_model_name}")
print(f"R²: {results[best_model_name]['r2']:.4f}")
print(f"MSE: {results[best_model_name]['mse']:.4f}")

# 最良モデルの詳細分析
if "標準化" in best_model_name:
    X_scaled = X_standard
    scaler = standard_scaler
else:
    X_scaled = X_robust
    scaler = robust_scaler

# statsmodelsによる詳細な統計分析
X_with_const = sm.add_constant(X_scaled)
model_sm = sm.OLS(y, X_with_const).fit()
print("\n詳細な統計情報:")
print(model_sm.summary())

# 予測値と実測値のプロット
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)

# 分布が広範囲の場合は95%のデータ範囲に制限
p95_max = np.percentile(y, 95)
plt.xlim(0, p95_max)
plt.ylim(0, p95_max)

plt.xlabel("実測値")
plt.ylabel("予測値")
plt.title(f"{best_model_name} - 実測値 vs 予測値")
plt.tight_layout()
plt.show()

# 特徴量の重要度の可視化（スケーリング前の単位に戻す）
if "標準化" in best_model_name:
    coeffs = best_model.coef_ / standard_scaler.scale_
else:
    coeffs = best_model.coef_ / robust_scaler.scale_

plt.figure(figsize=(10, 6))
plt.bar(X.columns, coeffs)
plt.xlabel("特徴量")
plt.ylabel("係数")
plt.title(f"{best_model_name} - 特徴量の重要度")
plt.tight_layout()
plt.show()

# 残差分析
# 最良モデルのテストデータにおける予測値
best_y_pred_all = best_model.predict(X_scaled)
residuals = y - best_y_pred_all

plt.figure(figsize=(10, 6))
plt.scatter(best_y_pred_all, residuals, alpha=0.5)
plt.hlines(
    y=0,
    xmin=best_y_pred_all.min(),
    xmax=best_y_pred_all.max(),
    colors="r",
    linestyles="--",
)
plt.xlabel("予測値")
plt.ylabel("残差")
plt.title("残差プロット")
plt.tight_layout()
plt.show()

# 残差の分布
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("残差")
plt.title("残差の分布")
plt.tight_layout()
plt.show()

# 改良されたモデルでの各特徴量とターゲットの関係
fig, axes = plt.subplots(
    2, 2, figsize=(16, 12), gridspec_kw=dict(wspace=0.1, hspace=0.3)
)
axes = axes.flatten()

# 各特徴量とPrice（外れ値処理済み）の散布図
for i, feature in enumerate(X.columns):
    axes[i].scatter(df_cleaned[feature], df_cleaned["Price"], alpha=0.1)

    # 線形トレンドラインを追加
    m, b = np.polyfit(df_cleaned[feature], df_cleaned["Price"], 1)
    axes[i].plot(df_cleaned[feature], m * df_cleaned[feature] + b, "r")

    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("価格 [円]")
    axes[i].set_title(f"カプセルトイ価格と{feature}")

plt.tight_layout()
plt.show()

# カテゴリと価格の関係
plt.figure(figsize=(14, 6))
sns.boxplot(x="Category", y="Price", data=df)
plt.xticks(rotation=90)
plt.ylim(0, 2100)
plt.title("カテゴリ別の価格分布")
plt.tight_layout()
plt.show()

# 日付と価格の関係（時系列トレンド）
plt.figure(figsize=(14, 6))
df_cleaned.groupby("Date")["Price"].mean().plot()
plt.title("日付ごとの平均価格推移")
plt.ylabel("平均価格")
plt.tight_layout()
plt.show()
