import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import urllib.request
from tqdm import tqdm
import os

def get_xs_links(url):
    http = httplib2.Http(disable_ssl_certificate_validation=True)
    s, r = http.request(url)
    xs_links = []
    soup = BeautifulSoup(r, 'html.parser')
    for link in soup.find_all('img'):
        tmp = link.get('src')
        if 'get' in tmp:
            xs_links.append(tmp)
    return xs_links

def get_links_from_yandex(search_url, total_pages):
    links = []
    for i in tqdm(range(total_pages)):
        page_url = search_url.format(i)
        xs_links = get_xs_links(page_url)
        orig_links = [link[:-2] + 'orig' for link in xs_links]
        links += orig_links
    return links


models1 = {
    'iPhone-4s': ('https://fotki.yandex.ru/search.xml?grouping=off&text=iPhone%204s&type=model&&p={}', 128),
    'iPhone-6': ('https://fotki.yandex.ru/search.xml?text=iPhone 6&type=model&&p={}', 128),
    'Sony-NEX-7' : ('https://fotki.yandex.ru/search.xml?text=Sony%20NEX%207&type=model&&p={}', 128),
    'Samsung-Galaxy-S4' : ('https://fotki.yandex.ru/search.xml?text=Samsung%20galaxy%20s4&type=model&&p={}', 128),
    'Samsung-Galaxy-Note3' : ('https://fotki.yandex.ru/search.xml?text=Samsung%20galaxy%20note%203&type=model&&p={}', 128),
    'Motorola-X' : ('https://fotki.yandex.ru/search.xml?text=Motorola%20Droid%20X&type=model&&p={}', 17),
    'Motorola-Nexus-6': ('https://fotki.yandex.ru/search.xml?text=Motorola%20Nexus%206&type=model&&p={}', 128),
    'Motorola-Droid-Maxx-1060' : ('https://fotki.yandex.ru/search.xml?text=Motorola%20XT1060&type=model&&p={}', 9),
    'Motorola-Droid-Maxx-1080' : ('https://fotki.yandex.ru/search.xml?text=Motorola%20XT1080&type=model&&p={}', 48)
}

models2 = {
    'Motorola-X-1052' : ('https://fotki.yandex.ru/search.xml?text=motorola%20XT%201052&type=model&&p={}', 43),
    'Motorola-X-1053' : ('https://fotki.yandex.ru/search.xml?text=motorola%20XT%201053&type=model&&p={}', 16),
    'Motorola-X-1055' : ('https://fotki.yandex.ru/search.xml?text=motorola%20XT%201055&type=model&&p={}', 2),
    'Motorola-X-1056' : ('https://fotki.yandex.ru/search.xml?text=motorola%20XT%201056&type=model&&p={}', 128),
    'Motorola-X-1058' : ('https://fotki.yandex.ru/search.xml?text=motorola%20XT%201058&type=model&&p=26', 27),
    
}

path = 'files/{}/'
for model, conf in models1.items():
    os.makedirs(path.format(model), exist_ok=True)
    links = get_links_from_yandex(*conf)
    with open(path.format(model) + model + '.txt', 'w') as out:
        for link in links:
            out.write(link + '\n')



path = 'files/{}/'
for model, conf in models2.items():
    os.makedirs(path.format(model), exist_ok=True)
    links = get_links_from_yandex(*conf)
    with open(path.format(model) + model + '.txt', 'w') as out:
        for link in links:
            out.write(link + '\n')






