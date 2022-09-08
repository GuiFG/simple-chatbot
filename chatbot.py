import random 
import pln
import game 
import topics
import log 


def output(input, topic):
    if topic is None:
        return 'Desculpe, mas nao entendi. Pode repetir ?'

    response = topics.get_response(topic, input)

    return response 


def run():
    print("(escreva 'sair' para finalizar)")
    print("Ola, sou um chatbot e vou responder perguntas sobre jogos: ")

    while True:
        user_input = input('Usuario: ')
        user_input = user_input.lower()

        if user_input == 'sair':
            log.write('Ate breve!')
            break 
        
        topic = topics.get_topic(user_input)
        response = output(user_input, topic)
        log.write(response)
        


