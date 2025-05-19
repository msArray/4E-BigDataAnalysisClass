import os
import pandas as pd
import sys
from pathlib import Path
import csv
import re


def csv_merge_with_avg():
    folder_paths = [
        "./b101_20250518015714",
        "./gs201612_20250518021159",
        "./gsb202102_20250518021133",
    ]
    """
    指定されたフォルダ内のすべてのCSVファイルをマージし、平均値を計算します。
    
    Args:
        folder_paths (list): CSV ファイルが含まれるフォルダのパスのリスト
    """
    # フォルダパスを確認
    for folder_path in folder_paths:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            print(
                f"エラー: フォルダ '{folder_path}' が存在しないか、ディレクトリではありません。"
            )
            return

    # CSVファイルを検索
    csv_files = []
    for folder_path in folder_paths:
        folder = Path(folder_path)
        csv_files.extend(folder.glob("*.csv"))

    if not csv_files:
        print("フォルダにCSVファイルが見つかりませんでした。")
        return

    print(f"{len(csv_files)}個のCSVファイルが見つかりました。マージを開始します...")

    # 各ファイルをマージ
    merged_data = pd.DataFrame(columns=["year", "month", "price"])

    for csv_file in csv_files:
        try:
            if csv_file.name[0] == "b":
                df = pd.read_csv(
                    csv_file.__str__(), encoding="utf-8", skiprows=10, usecols=[13]
                )
                content = []
                with open(csv_file.__str__(), encoding="utf-8") as f:
                    reader = csv.reader(f)
                    content = [row for row in reader]
                df.columns = ["price"]
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                date = content[10][8]
                t = re.split("[年月]", date)
                print(t[0], t[1])
                merged_data = pd.concat(
                    [
                        merged_data,
                        pd.DataFrame(
                            [
                                {
                                    "year": int(t[0]),
                                    "month": int(t[1]),
                                    "price": df["price"].mean(),
                                }
                            ]
                        ),
                    ]
                )
            elif "gsb" in csv_file.name:
                df = pd.read_csv(
                    csv_file.__str__(), encoding="utf-8", skiprows=8, usecols=[7]
                )
                content = []
                with open(csv_file.__str__(), encoding="utf-8") as f:
                    reader = csv.reader(f)
                    content = [row for row in reader]
                df.columns = ["price"]
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                date = content[8][1]
                t = re.split("[年月]", date)
                merged_data = pd.concat(
                    [
                        merged_data,
                        pd.DataFrame(
                            [
                                {
                                    "year": int(date[0:4]),
                                    "month": int(date[6:8]),
                                    "price": df["price"].mean(),
                                }
                            ]
                        ),
                    ]
                )
            elif "gs" in csv_file.name:
                df = pd.read_csv(
                    csv_file.__str__(), encoding="utf-8", skiprows=15, usecols=[2]
                )
                content = []
                with open(csv_file.__str__(), encoding="utf-8") as f:
                    reader = csv.reader(f)
                    content = [row for row in reader]
                df.columns = ["price"]
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                date = content[10][2]
                t = re.split("[{平成}年月]", date)
                t = [a for a in t if a != ""]
                for i in range(len(t)):
                    t[i] = t[i].lower()

                if int(t[0]) < 2000:
                    t[0] = int(t[0]) + 1988

                t[0] = int(t[0])
                t[1] = int(t[1])
                merged_data = pd.concat(
                    [
                        merged_data,
                        pd.DataFrame(
                            [
                                {
                                    "year": t[0],
                                    "month": t[1],
                                    "price": df["price"].mean(),
                                }
                            ]
                        ),
                    ]
                )
        except Exception as e:
            print(
                f"エラー: '{csv_file.name}' のマージ中にエラーが発生しました: {str(e)}"
            )

    # merged_dataをCSVファイルに保存
    merged_data["date_key"] = merged_data["year"] * 100 + merged_data["month"]
    merged_data = merged_data.sort_values(by='date_key')
    merged_data = merged_data.drop(columns=["date_key"])
    output_file = "merged_gasolin_merge.csv"
    merged_data.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"マージされたデータは '{output_file}' に保存されました。")


csv_merge_with_avg()
