import numpy as np
import pandas as pd
import httplib2
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm
import os


folders = os.listdir('html_pages')

for i in folders:
    right_links = []

    for page in os.listdir('html_pages/' + i):
        file = open('html_pages/'+i+'/'+page, 'r')
        
        html_doc = file.read() 

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_doc, 'html.parser')

        res = []
        for link in soup.find_all('a'):
            tmp = link.get('href')
            if tmp is not None:
                if 'photos' in tmp and 'comments' not in tmp and 'pool' in tmp:
                    res += [tmp]
        res = np.array(res)
        res = np.unique(res)

        res = pd.DataFrame(res)
        res[0] = res[0].apply(lambda x : x.split('in/pool')[0]+'sizes/o')

        links = res.values
                
        for url in tqdm(links):
            http = httplib2.Http()
            status, sec_html = http.request(url[0])

            soup = BeautifulSoup(sec_html, 'html.parser')
            for link in soup.find_all('img'):
                tmp = link.get('src')
                if 'c1.staticflickr' in tmp:
                    img_url = tmp
                    right_links += [img_url]
                    
    pd.DataFrame(right_links).to_csv('links/{}.csv'.format(i), index=False)