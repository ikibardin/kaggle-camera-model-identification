import numpy as np
import pandas as pd
import httplib2
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm
import os


folders = os.listdir('links/')
for i in filders:
    cnt = 0
    pd.read_csv('links/'+i)
    img = urllib.request.urlopen(img_url).read()
    out = open("files/{}/{}.jpg".format(i,cnt), "wb")
    out.write(img)
    out.close
    cnt += 1