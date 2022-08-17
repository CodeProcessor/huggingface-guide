import time

from transformers import pipeline


def text_classification():
    text_classification_list = [
        "This restaurant is awesome",
        "This restaurant is not awesome",
        "This service is great",
        "This service is poor",
    ]
    start_time = time.time()
    pipe = pipeline("text-classification")
    print("Model Loaded! in {} seconds".format(time.time() - start_time))

    for text in text_classification_list:
        start_time = time.time()
        print(text)
        print(pipe(text))
        print("Prediction took {} seconds".format(time.time() - start_time))


def question_and_answering():
    question = "What is python supports?"
    context = "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed."
    start_time = time.time()
    qna = pipeline("question-answering")
    print("Model Loaded! in {} seconds".format(time.time() - start_time))
    ret = qna(question, context)
    print(ret)


def text_generation():
    question = "Hello I am a"
    start_time = time.time()
    model = pipeline("text-generation")
    print("Model Loaded! in {} seconds".format(time.time() - start_time))
    ret = model(question)
    print(ret)


def token_classification():
    text = "Hello my name is Dulan. I lives in Colombo. I like to play cricket"
    start_time = time.time()
    model = pipeline("token-classification", model="Jean-Baptiste/camembert-ner")
    print("Model Loaded! in {} seconds".format(time.time() - start_time))
    ret = model(text)
    print(ret)


def sentiment_analysis():
    text = "This bike is great quality. Sure, it isn’t as light as a road bike...maybe because it isn’t a road bike. It’s a mountain bike and it feels strong and sturdy. Assembly was mindless and easy. Front and rear derailleur needed tuning upon first ride, which is normal. The brakes also had to be adjusted which was relatively easy done alone. 5 stars, especially at this great affordable price."
    start_time = time.time()
    model = pipeline("sentiment-analysis")
    print("Model Loaded! in {} seconds".format(time.time() - start_time))
    ret = model(text)
    print(ret)


if __name__ == '__main__':
    # text_generation()
    sentiment_analysis()
