import pln 
import game 
import log 
import random 
import re 

greetings_entry = ('olá', 'opa', 'oi', 'ola', 'eae', 'de boa', 'tudo certo', 'tudo bom', 'bom dia', 'boa tarde', 'boa noite', 'oba')
greetings_response = ('olá', 'opa', 'oi', 'bem-vindo', 'como você está?', 'tudo bem?')

def checkGreetings(text):
    grams = pln.n_gram_extractor_range(text, 1, 2)

    for gram in grams:
        if gram.lower() in greetings_entry:
            return True 

    return False

def greetings(text):
    grams = pln.n_gram_extractor_range(text, 1, 2)

    for gram in grams:
        if gram.lower() in greetings_entry:
            return random.choice(greetings_response) 

    return 'Nao entendi, pode repetir ?'

def checkReview(input):
    options = ['review', 'sobre', 'quero saber', 'informacoes', 'informacao']

    for option in options:
        if option in input:
            return True 
    
    return False  

def review(input):
    try:
        success, result = game.getGameName(input)
        if not success:
            update_active_topic('review')
            return result 

        gameName = result 
        log.write('recuperando a review do jogo: ' + gameName)
        review = game.getReview(gameName)

        short_review = pln.summarization(review, 5)

        log.debug('tentando traduzir a review')
        short_review = pln.translate_sentence(short_review)

        update_active_topic('')
        return short_review
    except Exception as e:
        log.debug(str(e))
        return 'Nao foi possivel obter a review'

def checkRecommendation(input):
    options = ['recomendacao', 'recomendacoes', 'semelhantes', 'semelhante']

    for option in options:
        if option in input:
            return True 
    
    return False 

def recommendation(input):
    try:
        success, result = game.getGameName(input)
        if not success:
            update_active_topic('recommendation')
            return result 

        game_name = result 
        log.write('recuperando recomendacoes semelhantes ao jogo: ' + game_name)

        info = game.get_info_games()
        print('info', info)
        model = pln.get_model_similarity(info)
        print('model', model)
        games = game.get_games_names_df()
        print('games', games)
        similar_games = pln.most_games_similarities(game_name, games, model)
        print('similar_games', similar_games)
        
        update_active_topic('')
        return 'Jogos semelhantes: ' + ', '.join(similar_games)
    except Exception as e:
        log.debug(str(e))
        return 'Nao foi possivel obter a recomendacao'
 
def checkFeedback(input):
    options = ['feedback', 'opinioes', 'percepcao']

    for option in options:
        if option in input:
            return True 
    
    return False 

def feedback(input):
    try:
        success, result = game.getGameName(input)
        if not success:
            update_active_topic('feedback')
            return result 

        game_name = result 
        log.write('buscando os feedbacks do jogo: ' + game_name)
        feedbacks = game.getFeedbacks(game_name)
        
        text = ' '.join(feedbacks)
        result = pln.sentiment_analysis(text)

        update_active_topic('')
        return 'opiniao ' + result + ' do jogo'
    except Exception as e:
        log.debug(str(e))
        return 'Nao foi possivel obter os feedbacks do jogo'

def checkQuestionAnswer(input):
    return '?' in input

def questionAnswer(input):
    try:
        success, result = game.getGameName(input)
        if not success:
            update_active_topic('question_answer')
            update_input(input)
            return result 

        old_input = get_active_input()
        if old_input != '':
            input = old_input

        game_name = result 
        log.write('encontrando a resposta da pergunta = ' + input)
        answer = pln.question_answer(input, game_name)

        answer = re.sub(r"\[[0-9]*\]", '', answer)
        answer = answer.strip()

        update_input('')
        if answer is None:
            return 'Nao sei responder a pergunta :('

        return answer 
    except Exception as e:
        log.debug(str(e))
        return 'Nao foi possivel obter a resposta da pergunta'
    
    
topics = { 
    'greetings' : (checkGreetings, greetings), 
    'recommendation' : (checkRecommendation, recommendation),
    'feedback' : (checkFeedback, feedback),
    'review' : (checkReview, review),
    'question_answer' : (checkQuestionAnswer, questionAnswer),
    'active' : '',
    'input' : ''
}

def update_active_topic(topic):
    topics['active'] = topic

def get_active_topic():
    return topics['active']

def update_input(input):
    topics['input'] = input 

def get_active_input():
    return topics['input']

def get_topic(input):
    active_topic = get_active_topic()
    if active_topic != '':
        return active_topic

    for key, value in topics.items():
        if isinstance(value, str):
            continue

        check = value[0]
        if check(input):
            return key
        
    return None 

def get_response(topic, input):
    response = topics[topic][1]
    return response(input)


