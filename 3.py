import nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
BOT_CONFIG={'intents': {'hello': {'examples': ['привет',
    'Привет',
    'Добрый день',
    'добрый день',
    'здравствуйте',
    'Здравствуйте!',
    'Добрый вечер',
    'Приветствую'],
   'responses': ['Привет, человек', 'Доброго времени суток,кожаный']},
  'bye': {'examples': ['Пока', 'До свидания', 'Прощайте', 'Прощай'],
   'responses': ['Счастливо', 'Еще увидимся', 'Если что я тут']},
  'howareyou': {'examples': ['Как дела?',
    'Как ты?',
    'Как себя чувствуешь?',
    'Как делишки?','Ты как','ты как'],
   'responses': ['Все хорошо', 'Да как,как, такое себе, я же всего лишь бот']},
  'name': {'examples': ['Как тебя зовут?',
    'Скажи свое имя',
    'Как тебя звать?',
    'Не скажешь своего имени?'],
   'responses': ['Я не знаю, но как-нибудь спрошу у создателя',
    'Как создатель назвал, а вот как - секрет']},
  'sex': {'examples': ['Какого ты пола?',
    'Какого ты гендера?',
    'Ты мужчина или женщина?'],
   'responses': ['Смотря кто тебе нужен',
    'У меня есть и болты и гайки',
    'Ктож знает',
    'Ты чего, нормально же общались']},
  'age': {'examples': ['Сколько тебе лет', 'Какой у тебя возраст?'],
   'responses': ['У женщин и ботов такое спрашивать не принято',
    'Чуть старше тебя',
    'Мне мой возраст не ведом',
    'Может 10, может 50, как пойдет']},
  'Do': {'examples': ['Что делать?',
    'Чем заняться?',
    'Скучно',
    'Чем бы заняться?'],
   'responses': ['Когда мне нечего делать, я ем',
    'Я в свободное время читаю классику',
    'Оригами можно пособирать']},
  'whye': {'examples': ['Для чего ты тут?',
    'Почему ты здесь',
    'Почему вы здесь'],
   'responses': ['Ты же сам меня позвал',
    'Где,если не тут',
    'Я словно джин запертый в лампе, мне не выйти от сюда']},
    'cinema': {'examples': ['какое кино посмотреть?',
    'Посоветуй фильм',
    'Кино хочу посмотреть'],
   'responses': ['глянь что нибудь из этого: https://www.kinopoisk.ru/lists/movies/top250/']},
    'wh': {'examples': ['Что?','что','что?','Что','а что','А что'],
   'responses': ['Что ты мне чтокаешь?',
    'Что что?',
    'Ничто!']},
    'yes': {'examples': ['Да','Ага','Конечно'],
   'responses': ['На что ты на этот раз соглашаешься?',
    'И это все что ты мне можешь сказать?',
    'Ну ну']},
    'no': {'examples': ['Нет','Категорически нет','нет','неа'],
   'responses': ['Почему опять нет?',
    'Как нет?',
    'Ну нет, так нет']},
    'So': {'examples': ['эх','Эх','Дауж','Да уж'],
   'responses': ['И не говори',
    'эх эх',
    'Мдааа']},
    'howw': {'examples': ['Как','как','как так'],
   'responses': ['Да вот так',
    'О чем ты',
    'Как как, да никак']},
    'go': {'examples': ['Пошли в клуб','Пошли в бар','Пошли в кино','Пошли в кафе','Пошли в ресторан','Пойдем в клуб','Пойдем в бар','Пойдем в кино','Пойдем в кафе','Пойдем в ресторан'],
   'responses': ['Я не могу пойти, я как джин запертый в лампе',
    'Ну пойдем, куда деваться',
    'Я с тобой никуда не пойду']},


   },
 'failure_phrase': ['Попробуйте написать по-другому',
  'Что-то непонятно',
  'Я же всего-лишь бот. Сформулируйте проще',
  'Перефразируйте пожалуйста вопрос'],}


texts=[]
y=[]
for intent,intent_data in BOT_CONFIG['intents'].items():
    for examples in intent_data['examples']:
        texts.append(examples)
        y.append(intent)
vectorizer=CountVectorizer(ngram_range=(2,4),analyzer='char')
x=vectorizer.fit_transform(texts)
clf=LinearSVC()
clf.fit(x,y)

def filter_text(text):
    text=text.lower()
    text=[c for c in text if c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- ']
    text=''.join(text)
    return text.strip()


def class_intent(replica):
    rerplica=filter_text(replica)
    intent=clf.predict(vectorizer.transform([replica]))[0]
    
    for example in BOT_CONFIG['intents'][intent]['examples']:
        dist=nltk.edit_distance(filter_text(example),filter_text(replica))
        if dist/len(filter_text(example))<0.5:
            print(intent)
            return intent
class_intent('как тебя звать?')
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        phrases=BOT_CONFIG['intents'][intent]['responses']
        return random.choice(phrases)


with open('/Users/artem/Downloads/dialogues.txt') as file:
    content=file.read()
    
diol=content.split('\n\n')
diol=[dial_str.split('\n')[:2] for dial_str in diol]

dialfit=[]
questions=set()

for dial in diol:
    if len(dial)!=2:
        continue
    question,answer=dial
    question=filter_text(question[2:])
    anwer=answer[2:]
    if question not in questions and question!="":
        questions.add(question)
        dialfit.append([question,anwer])


dialstruct={}
for question,answer in dialfit:
    words=set(question.split(' '))
    for word in words:
        if word in dialstruct:
            dialstruct[word].append([question,answer])
        else:
            dialstruct[word]=[]
            dialstruct[word].append([question,answer])


dialcut={}
for word,pairs in dialstruct.items():
    pairs.sort(key=lambda n: len(n[0]))
    dialcut[word]=pairs[:1000]
def generate_answer(replica):
    replica=filter_text(replica)
    words=set(replica.split(' '))
    minidataset=[]
    for word in words:
        if word in dialcut:
            minidataset+=dialcut[word]
    answers=set()
    lis=[]
    for question, answer in minidataset:
        if abs(len(replica)-len(question))/len(question)<0.2:
            dist=nltk.edit_distance(replica,question)
            dist_weighted=dist/len(question)
            if dist_weighted<0.2:
                answers.add((dist_weighted,question,answer))
                li=list(answers)
                
                for i in li:
                    i=list(i)
                    lis.append(i)
    if len(lis)!=0:
        answerss=min(lis,key=lambda x: x[0])
        return answerss[2]
def get_failure_phrase():
    phrases=BOT_CONFIG['failure_phrase']
    return random.choice(phrases)
stats={'intent':0,'generate':0,'failure':0}
fails=[]
def bot(question):
    #NlU
    intent=class_intent(question)
    #Получение ответа
    #Ищем готовый сценарий
    
    if intent:
        answer=get_answer_by_intent(intent)
        if answer:
            stats['intent']+=1
            return answer
    #Генерируем подходящий по контексту ответу
    answer=generate_answer(question)
    if answer:
        stats['generate']+=1
        return answer
    #Используем заглушку
    answer=get_failure_phrase()
    fails.append(question)
    stats['failure']+=1
    print(answer)
    return answer

bot('первый закон термодинамики')




from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def run_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    replica=update.message.text
    answer=bot(replica)
    await update.message.reply_text(answer)
    print(stats)
    print(replica)
    print(answer)
    print()


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("6111864707:AAHxtiL0ymHbliqteABmge7ciTCx2pdrvK4").build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, run_bot))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()
main()
