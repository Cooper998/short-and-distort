import requests
import time
from bs4 import BeautifulSoup
import csv

url_base = 'https://www.tradingview.com/ideas/search/short%20bitcoin/page-'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}

with open("D:\\test.csv","a",newline='', encoding = 'utf-8') as csvfile: 
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Time","Author","Headline","Target"])
    #writer = csv.writer(csvfile)
    for i in range(1, 100):
        url = url_base + str(i)
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "lxml")

        card_items = soup.find_all('div', attrs={'class': 'tv-feed-layout__card-item'})

        for card_item in card_items:
            info = card_item.find('div', attrs={'class': 'tv-widget-idea__info-row'}).text.strip()
            if "Short" in info:
                print('Short')
                title = card_item.find('div', attrs={'class': 'tv-widget-idea__title-row'}).text.strip()
                try:
                    target = card_item.find('a',class_="tv-widget-idea__symbol apply-overflow-tooltip").text.strip()
                except:
                    target = None
                author = card_item.find('div', attrs={'class': 'tv-widget-idea__author-row'}).text.strip()
                t = card_item.find('span', attrs={'class': 'tv-card-stats__time'}).attrs['data-timestamp']
                timeStamp = int(float(t))
                timeArray = time.localtime(timeStamp)
                mytime = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)
                
                print(mytime)
                #print(time.strftime("%Y-%m-%d %H:%M:%S", timeArray))
                print(author)
                print(title)
                #print(info)
                print(target)
                print('\n')
                mylist = [mytime,author,title,target]
                writer.writerow(mylist)
    csvfile.close()





