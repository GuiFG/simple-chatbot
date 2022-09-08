import re
import nltk
import string
from collections import Counter 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np
from transformers import pipeline 
from urllib.parse import quote_plus
import urllib
from bs4 import BeautifulSoup
import spacy 
import googletrans

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('pt_core_news_sm')
stopwords = nltk.corpus.stopwords.words('english')
stopwords_pt = spacy.lang.pt.stop_words.STOP_WORDS

ner = spacy.load('en_core_web_sm')

sentiment_task = pipeline("sentiment-analysis")

URL_WIKIPEDIA = 'https://pt.wikipedia.org'

def remove_stop_words(tokens, language = 'en'):
    stop_words = stopwords
    if language == 'pt':
        stop_words = stopwords_pt

    new_text = []
    for token in tokens:
        if token not in stop_words and token not in string.punctuation:
            new_text.append(token)

    return new_text

def pre_processing(text, stopwords = 'en', lemma = True):
    text = text.lower()

    text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)

    text = re.sub(r" +", ' ', text)

    tokens = []
    for token in nltk.word_tokenize(text):
        tokens.append(token)

    new_text = remove_stop_words(tokens, stopwords)

    text = ' '.join([str(element) for element in new_text])

    if lemma:
        doc = nlp(text)
        list = []
        for token in doc:
            list.append(token.lemma_)
        
        text = ' '.join([str(element) for element in list])

    return text 

def score_words(freq_dist):
    max_freq = max(freq_dist.values())

    for word in freq_dist.keys():
        freq_dist[word] = (freq_dist[word]/max_freq)
    
    return freq_dist

def score_sentences(sents, freq_dist, max_len=30):
    sent_scores = {}

    for sent in sents:
        words = sent.split(' ')
        for word in words:
            if word.lower() not in freq_dist.keys() or len(words) >= max_len:
                continue

            if sent in sent_scores.keys():
                sent_scores[sent] += freq_dist[word.lower()]
            else:
                sent_scores[sent] = freq_dist[word.lower()]
    
    return sent_scores
                
def summarization(text, k):
    clean_text = re.sub('\n', '', text)
    clean_text = re.sub('-', '', clean_text)
    clean_text = pre_processing(text)

    word_tkn = nltk.word_tokenize(clean_text)
    freq_dist = nltk.FreqDist(word_tkn)
    freq_dist = score_words(freq_dist)

    sent_tkn = nltk.sent_tokenize(text)
    sent_scores = score_sentences(sent_tkn, freq_dist)

    top_sents = Counter(sent_scores)
    tops = top_sents.most_common(k)

    summary = ''
    for top in tops:
        summary += top[0].strip() + ' '

    return summary[:-1]

def get_similarity(sents_pre_process):
    tfidf = TfidfVectorizer()
    words_vectors = tfidf.fit_transform(sents_pre_process)

    similarity = cosine_similarity(words_vectors[-1], words_vectors)

    return similarity

def get_most_similar(content, text):
    sents = nltk.sent_tokenize(content)

    sents_pre_process = []
    for i in range(len(sents)):
        sents_pre_process.append(pre_processing(sents[i], stopwords='pt', lemma=True))

    text = pre_processing(text, stopwords='pt', lemma=True)
    sents_pre_process.append(text)

    similarity = get_similarity(sents_pre_process)

    sent_idx = similarity.argsort()[0][-2]
    similar_vector = similarity.flatten()
    similar_vector.sort()
    vector_found = similar_vector[-2]

    if vector_found != 0:
        print(sents)
        return sents[sent_idx]
    
    return None 

def get_model_similarity(doc):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tf.fit_transform(doc)
    doc_sim = cosine_similarity(tfidf_matrix)
    doc_sim_df = pd.DataFrame(doc_sim)

    return doc_sim_df

def most_games_similarities(game_name, games, doc_sims, top = 5):
    print('game_name', game_name)
    print('games', len(games))
    game_idx = np.where(games == game_name.lower())[0][0]
    print('game_idx', game_idx)
    game_similarities = doc_sims.iloc[game_idx].values 
    similar_game_idxs = np.argsort(-game_similarities)[1:top + 1]
    similar_games = games[similar_game_idxs]

    return similar_games

def get_label_sentiment_by_score(score):
    label = ''
    if score > 0.8:
        label = 'muito boa'
    elif score > 0.4:
        label = 'boa'
    elif score > 0:
        label = 'razoavel'
    elif score > -0.4:
        label = 'ruim'
    elif score > -0.7:
        label = 'pessima'
    else:
        label = 'horrivel'

    return label 

def sentiment_analysis(text):
    text = summarization(text, 512)

    sents = nltk.sent_tokenize(text)
    sentiments = sentiment_task(sents)

    score = 0
    count = 0
    for sentiment in sentiments:
        label = sentiment['label'].lower()
        if label == 'positive':
            score += sentiment['score']
        else:
            score -= sentiment['score']
        count += 1
    
    score = score / count 

    return get_label_sentiment_by_score(score)

def get_wiki_content_by_topic(topic):
    safe_string = quote_plus(topic)

    link = f'https://pt.wikipedia.org/w/index.php?search={safe_string}&title=Especial:Pesquisar&profile=advanced&fulltext=1&searchengineselect=mediawiki&ns0=1'
    
    dados = urllib.request.urlopen(link)
    dados_html = BeautifulSoup(dados, 'lxml')

    ul = dados_html.select('#mw-content-text > div.searchresults > ul')[0]
    li = ul.find_all('li')[0]
    href = li.find('a').attrs['href']

    link = URL_WIKIPEDIA + href 
    dados = urllib.request.urlopen(link)
    dados_html = BeautifulSoup(dados, 'lxml')
    paragraphs = dados_html.find_all('p')
    content = ''
    for p in paragraphs:
        content += p.text 
    content = content.lower()

    return content

def question_answer(text, topic):
    content = get_wiki_content_by_topic(topic)
    
    return get_most_similar(content, text)

def n_gram_extractor(sentence, n):
    grams = []
    tokens = re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
    for i in range(len(tokens)-n+1):
        g = ' '.join(tokens[i:i+n])
        grams.append(g)

    return grams 

def n_gram_extractor_range(sentence, start, end):
    grams = []
    if start == end:
        return n_gram_extractor(sentence, start)

    for i in range(start, end + 1):
        g_n = n_gram_extractor(sentence, i)
        grams.extend(g_n)

    return grams 

def translate_sentence(sentence, source = 'en', destination = 'pt'):
    translator = googletrans.Translator()

    result = translator.translate(sentence, src=source, dest=destination)

    return result.text