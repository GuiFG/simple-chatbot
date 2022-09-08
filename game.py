
import urllib.request as urllib2
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import csv
import pln 
import pandas as pd
import numpy as np 
import re
import json 
import nltk

URL_GAMESPOT = "https://www.gamespot.com"

def getSoup(link):
    response = urllib2.urlopen(link)
    html_doc = response.read()
    with open('index.html', 'w') as f:
        f.write(str(html_doc))
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    return soup

def getLinkReview(name):
    safe_string = quote_plus(name)
    url = URL_GAMESPOT + f'/search/?q={safe_string}&i=reviews'
    soup = getSoup(url)

    list = soup.select('#js-sort-filter-results')[0]
    game = list.find('li')
    link_review = game.find('a')

    return URL_GAMESPOT + link_review.get('href')

def getReview(name):
    link_review = getLinkReview(name)
    soup = getSoup(link_review)

    article = soup.find('article')
    content = article.select('div.js-content-entity-body')[0]
    paragraphs = content.find_all('p')
    
    review = ''
    for paragraph in paragraphs:
        review += paragraph.getText()
    
    return review 

def get_item_by_class(list, class_name):
    result = ''
    for item in list:
        try:
            classes = item.attrs['class']
            cls = ' '.join(classes)
            if class_name in cls:
                result = item
                break 
        except:
            result = ''
    
    return result 

def get_item_by_attr_contains(list, attr, value):
    result = ''
    for item in list:
        try:
            atr = item.attrs[attr]
            if value in atr:
                result = atr
                break 
        except:
            result = ''

    return result 

def get_list_item_cotains(list, attr, value):
    result = []
    count = 0
    for item in list:
        try:
            atr = item.attrs[attr]
            if value in atr:
                result.append(item)
        except:
            count += 1

    return result 

def get_html_comments(soup):
    article = soup.find('article')
    links = article.find_all('a')
    comment_id = get_item_by_attr_contains(links, 'href', 'comments')
    
    match = re.findall(r'[0-9]', comment_id)
    id = ''.join(match)

    link_comments = URL_GAMESPOT  + f'/forums/comments/{id}/?subTopic=0&comment_page=1&wrap=1'
    response = urllib2.urlopen(link_comments)
    response_json = response.read()
    response_json = response_json.decode('ascii')
    response_json = json.loads(response_json)
    
    return response_json['html']

def get_paragraphs_comments(div_comments):
    result = []
    for div in div_comments:
        article = div.find('article')
        paragraphs = article.find_all('p')
        for p in paragraphs:
            result.append(p.getText())
    
    return result 

def getFeedbacks(name):
    link_review = getLinkReview(name)
    print(link_review)
    soup = getSoup(link_review)
    html = get_html_comments(soup)
    soup = BeautifulSoup(html, 'html.parser')

    divs = soup.find_all('div')
    div_comments = get_item_by_class(divs, 'comment-messages')
    div_comments = div_comments.find_all('div')
    div_comments = get_list_item_cotains(div_comments, 'id', 'js-message')
    comments = get_paragraphs_comments(div_comments)

    return comments 
    
def get_games_names():
    file = open('./dados/video_game_dataset.csv')
    csvReader = csv.reader(file)
    data = list(csvReader)

    names = []
    for item in data:
        names.append(item[1])

    return names 

def get_games_names_df():
    df = pd.read_csv('./dados/video_game_dataset.csv')

    names = df['Name'].values
    lower = lambda x: x.lower()
    vfunc = np.vectorize(lower)
    return vfunc(names)

def get_info_games():
    df = pd.read_csv('./dados/video_game_dataset.csv')

    df['info'] = df['Platform'] + ' ' + df['Genre']
    info = list(df['info'])

    return info

def get_most_similar(words, value):
    score = 0
    idx = 0
    for i, word in enumerate(words):
        sents = word.split(' ')
        sents.append(value)
        similarity = pln.get_similarity(sents)
        aux = similarity[0].sum()

        if aux > score:
            idx = i 
            score = aux 

    return (words[idx], score)

def named_game_recognition(text):
    text = pln.pre_processing(text)

    tokens = nltk.word_tokenize(text)

    tokens = pln.remove_stop_words(tokens, 'pt')

    text = ' '.join(tokens)
    tokens = pln.n_gram_extractor_range(text, 1, 2)

    games = get_games_names()
    max_score = 0
    game_name = ''
    for token in tokens:
        name, score = get_most_similar(games, token)
        print(name, score)
        if score > max_score:
            max_score = score
            game_name = name 
    
    if max_score <= 1:
        return None 

    return game_name

def getGameName(text):
    name = named_game_recognition(text)
    
    if name is None:
        return (False, 'Nao reconheci o nome do jogo. Pode repetir o nome dele ?')
    
    return (True, name)