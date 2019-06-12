from bs4 import BeautifulSoup, SoupStrainer
import urllib.request as ur
import re

visited = []

def crawler(s):
    print( " s = " + s)
    visited.append(s)
    response = ur.urlopen(s)
    html_doc = response.read()

    soup = BeautifulSoup(html_doc, 'html.parser')
    urlList = []


    for link in soup.findAll('a', attrs= {'href': re.compile("^http://")}):
        with open("urlList.txt" , 'a') as f:
            if link.has_attr('href'):            
                y = (link.get('href')) 
                urlList.append(y)
            print(urlList)
            f.write(y)

    # with open("urlList.txt", 'r') as x:
    #     for url in x:
    #         if url not in visited:
    #             crawler(str(url))




crawler("https://arstechnica.com")