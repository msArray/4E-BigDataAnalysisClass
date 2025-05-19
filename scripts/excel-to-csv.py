import os
import pandas as pd
import sys
from pathlib import Path

def convert_excel_to_csv(folder_path):
    """
    指定されたフォルダ内のすべてのExcelファイル(.xlsx, .xls)をCSVに変換します
    
    Args:
        folder_path (str): Excel ファイルが含まれるフォルダのパス
    """
    # フォルダパスを確認
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"エラー: フォルダ '{folder_path}' が存在しないか、ディレクトリではありません。")
        return
    
    # Excelファイルを検索
    excel_files = []
    for ext in ['.xlsx', '.xls', '.xlsm']:
        excel_files.extend(folder.glob(f'*{ext}'))
    
    if not excel_files:
        print(f"フォルダ '{folder_path}' にExcelファイルが見つかりませんでした。")
        return
    
    print(f"{len(excel_files)}個のExcelファイルが見つかりました。変換を開始します...")
    
    # 各ファイルを変換
    success_count = 0
    error_count = 0
    
    for excel_file in excel_files:
        try:
            # 各シートを個別のCSVファイルに変換
            xlsx = pd.ExcelFile(excel_file)
            sheet_names = xlsx.sheet_names
            
            # シートが1つだけの場合は単一のCSVファイルを作成
            if len(sheet_names) == 1:
                csv_path = excel_file.with_suffix('.csv')
                df = pd.read_excel(excel_file, sheet_name=sheet_names[0])
                df.to_csv(csv_path, encoding='utf-8', index=False)
                print(f"変換成功: {excel_file.name} -> {csv_path.name}")
                success_count += 1
            # 複数のシートがある場合は、シート名を付与したCSVファイルを作成
            else:
                for sheet in sheet_names:
                    sheet_name_safe = sheet.replace('/', '_').replace('\\', '_')
                    csv_path = excel_file.with_name(f"{excel_file.stem}_{sheet_name_safe}.csv")
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    df.to_csv(csv_path, encoding='utf-8', index=False)
                    print(f"変換成功: {excel_file.name} (シート: {sheet}) -> {csv_path.name}")
                    success_count += 1
                
        except Exception as e:
            print(f"エラー: '{excel_file.name}' の変換中にエラーが発生しました: {str(e)}")
            error_count += 1
    
    print(f"\n変換完了: 成功 {success_count} 件, エラー {error_count} 件")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # コマンドライン引数からフォルダパスを取得
        folder_path = sys.argv[1]
    else:
        # 引数がない場合は現在のディレクトリを使用
        folder_path = os.getcwd()
        print(f"フォルダパスが指定されていないため、現在のディレクトリを使用します: {folder_path}")
    
    convert_excel_to_csv(folder_path)