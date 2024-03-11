from datetime import datetime
import csv
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from functools import partial

from .news_category import NewsCategory
from .format_time import get_formatted_current_date_time, get_formatted_current_date, format_date

MAX_PAGE = 11

# TODO: 시간 단위로 수집했던 기사는 다시 접근하지 않기 (dict() {"society" : "23-11-05 18:32"})
class NewsScraper:
    def __init__(self):
        self.results = []

    def scrape_url_per_category(self, category):
        articles = []
        for page in range(1, MAX_PAGE):
            page_url = f"https://news.daum.net/breakingnews/{category}?page={page}"
            response = requests.get(page_url)

            if response.status_code != 200:
                print(f"\033[36m[Mini MLOps] \033[91m{page_url}를 불러오는 데 문제가 발생했습니다")
                print(f"\033[36m[Mini MLOps] \033[91m{category.upper()} PROCESS POOL을 비정상 종료합니다.\n")
                return

            print(f" ######  {category.upper()}, {page}페이지 스크래핑 중입니다.")
            news_list_html = BeautifulSoup(response.text, "html.parser")
            url_list = news_list_html.find("ul", class_="list_news2 list_allnews")

            if url_list is None:
                print(f"\033[36m[Mini MLOps] \033[91m{category.upper()} {page}페이지는 존재하지 않습니다.")
                print(f"\033[36m[Mini MLOps] \033[91m{category.upper()} PROCESS POOL을 종료합니다.\n")
                break

            urls = url_list.find_all("a", class_="link_txt")
            
            if not urls:
                print(f"\033[36m[Mini MLOps] \033[91m{category.upper()} {page}페이지는 존재하지 않습니다.")
                print(f"\033[36m[Mini MLOps] \033[91m{category.upper()} PROCESS POOL을 종료합니다.\n")
                break

            articles.extend(self.scrape_articles_from_urls(urls, category))
        
        return articles
    
    def scrape_less_per_category(self, category):
        articles = []

        page_url = f"https://news.daum.net/breakingnews/{category}"
        response = requests.get(page_url)

        if response.status_code != 200:
            print(f"\033[36m[Mini MLOps] \033[91m{page_url}를 불러오는 데 문제가 발생했습니다")
            print(f"\033[36m[Mini MLOps] \033[91m{category.upper()} PROCESS POOL을 비정상 종료합니다.\n")
            return

        print(f"\033[36m[Mini MLOps] \033[37m{category.upper()} 스크래핑 중입니다.")
        news_list_html = BeautifulSoup(response.text, "html.parser")
        url_list = news_list_html.find("ul", class_="list_news2 list_allnews")

        if url_list is None:
            # print(f"\t- {category.upper()} {page}페이지는 존재하지 않습니다.")
            print(f"\033[36m[Mini MLOps] \033[37m{category.upper()} PROCESS POOL을 종료합니다.\n")
            return

        urls = url_list.find_all("a", class_="link_txt")
        
        if not urls:
            print(f"\033[36m[Mini MLOps] \033[37m{category.upper()} PROCESS POOL을 종료합니다.\n")
            return

        articles.extend(self.scrape_articles_from_urls(urls, category))
        
        return articles

    def scrape_articles_from_urls(self, urls, category):
        articles = []

        for index, url in enumerate(urls):
            print(f"\033[36m[Mini MLOps] \033[37m{category.upper()} {index + 1}번째 기사, {url['href']}를 성공적으로 가져왔습니다.")
            scraped_article = self.scrape_article_from_url(url["href"], category)
            articles.append(scraped_article)

        return articles

    def scrape_article_from_url(self, url, category):
        response = requests.get(url)

        if response.status_code != 200:
            print(f"{url}를 불러오는 데 문제가 발생했습니다")
            print(f"\033[36m[Mini MLOps] \033[37m{category.upper()} PROCESS POOL을 종료합니다.\n")
            return

        news_html = BeautifulSoup(response.text, "html.parser")
        current_time = get_formatted_current_date_time()
        upload_time = news_html.find("span", class_="num_date").text
        title = news_html.find("h3", class_="tit_view").text
        content = news_html.find("div", class_="article_view").text.strip()

        return {
            "category": category,
            "title": title,
            "content": content,
            "upload_datetime": format_date(upload_time)
        }

    def first_run(self):
        print("\033[32m첫 뉴스 스크래핑을 시작합니다.\n")
        news_categories = [category.value for category in NewsCategory]

        with Pool(processes=len(news_categories)) as pool:
            results = pool.map(self.scrape_url_per_category, news_categories)
        
    def run(self):
        print("\033[32m뉴스 스크래핑을 시작합니다.\n")
        news_categories = [category.value for category in NewsCategory]

        with Pool(processes=len(news_categories)) as pool:
            results = pool.map(self.scrape_less_per_category, news_categories)
        
        return results

if __name__ == "__main__":
    news_scraper = NewsScraper()
    news_scraper.run()
