import requests
from bs4 import BeautifulSoup
import urllib.request 
import csv
import re
import time

# URLからHTMLをダウンロードし、テキストを抽出してファイルに保存するスクリプト
def fetch_html(url: str) -> str:
    with urllib.request.urlopen(url) as res:
        html = res.read().decode()
    return html

# テキストの中から空白の行を削除する関数
def remove_blank_lines(text):
    lines = text.split('\n')  
    non_blank_lines = [line for line in lines if line.strip()]  
    return '\n'.join(non_blank_lines) 

# CSVファイルからURLを抽出する関数
def create_url_list(csv_file,target_column_index):
    url_list = []
    with open(csv_file, 'r', encoding='SHIFT_JIS') as file:
        csv_reader = csv.reader(file)    
        for row in csv_reader:
            if len(row) > target_column_index:  # 行が指定した列を持つか確認
                cell_value = row[target_column_index]
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', cell_value)
                if urls:
                    url_list.extend(urls)
    return url_list


def download_and_process_urls(url_list):
    documents = []  # 各URLのテキストを格納するリスト
    print(f"URLの数: {len(url_list)}")
    for url in url_list:
        time.sleep(5)  # 5秒待機（サーバーへの負荷を軽減するため）
        print(f"URL: {url}")
        
        try:
            # URLからHTMLをダウンロード
            html = fetch_html(url)
            print(f"URL {url} のHTMLをダウンロードしました。")

            # BeautifulSoupで解析（必要に応じて行う）
            soup = BeautifulSoup(html, 'html.parser')
            
            
            # 5週分の空のデータ
            # 週に合わせてデータの格納
            data = [[] for _ in range(5)]
            # data = [[] for _ in range(3)]
            global month
            
            "2017年 6月と7月を境にページの構造が変わったため6月以前を追加する必要あり"
            
            # 2017年 7月以降対応レイアウト
            
            weekly_data: list[BeautifulSoup] = soup.find_all("div", class_="week")
            for weekly in weekly_data:
                t = weekly.find_all("span", class_="pg-schedule__month--date")
                if t is None or len(t) == 0:
                    week = 0
                else:
                    month = int(t[0].get_text()) 
                    week = int(t[1].get_text())
                weekly_items = weekly.find_all("div", class_="c-card__list")
                resolved_items = []
                for weekly_item in weekly_items:
                    name = weekly_item.find("p", class_="c-card__name").get_text()
                    price = weekly_item.find("span", class_="c-card__price--main").get_text()
                    category = weekly_item.find("i", class_="c-card__category").get_text()
                    resell = weekly_item.find("span", class_="c-card__resale--txt")
                    is_resell = False
                    if resell is not None:
                        is_resell = True if resell.get_text() == "再入荷" else False
                    resolved_items.append([name, price, category, is_resell])
                    print([name, price, category, is_resell])
                
                data[week - 2] = resolved_items
            
            # 2017年 6月以前対応レイアウト
            # weekly_data: list[BeautifulSoup] = soup.find_all("div", class_="week")
            # f = 0
            # for weekly in weekly_data:
            #     t = weekly.find_all("span", class_="pg-schedule__month--tbd")
            #     n = soup.find_all("span", class_="pg-tit__main--mo")
            #     if t is None or len(t) == 0:
            #         week = 0
            #     else:
            #         month = int(n[0].get_text().split('.')[1]) 
            #         week = t[0].get_text()
            #         
            #     weekly_items = weekly.find_all("div", class_="item")
            #     resolved_items = []
            #     for weekly_item in weekly_items:
            #         name = weekly_item.find("div", class_="title").get_text()
            #         price = weekly_item.find("div", class_="price").get_text().split("円")[0]
            #         category = weekly_item.find("div", class_="category").get_text()
            #         is_resell = False
            #         resolved_items.append([name, price, category, is_resell])
            #         print([name, price, category, is_resell])
            #     
            #     data[f] = resolved_items
            #     f += 1
            #     
            print(data)
            # データ構造
            "月 週"
            "名前 価格 カテゴリー 再販か否か"
            
            for i, wk in enumerate(data):
                # テキストをファイルに保存
                # 命名則 年月_週.txt
                with open(f'./scraped_before/{url.split("=")[1]}_{i + 1}.csv', 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['月','週'])
                    writer.writerow([month, i + 1])
                    writer.writerow(['名前', '価格', 'カテゴリー', '再販'])
                    for item in wk:
                        writer.writerow(item)
                    print(f"URL {url} のデータをファイルに保存しました。")                
                

        except urllib.error.URLError as e:
            print(f"URL {url} からのダウンロードエラー: {e}")
        except Exception as e:
            print(f"URL {url} の処理中にエラーが発生しました: {e}")
    
    return documents

def save_documents_as_text_file(documents, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for document in documents:
            f.write(document + '\n') 


def main():
    
    # CSVファイルとURLのある列を指定
    url_list = create_url_list('./URL_list_plus.csv', 0) 

    # 各URLからHTMLをダウンロードしてテキストに変換
    download_and_process_urls(url_list)
    
    # documentsを一つのテキストファイルとして保存
    # save_documents_as_text_file(documents, './output.txt')

if __name__ == "__main__":
    main()
