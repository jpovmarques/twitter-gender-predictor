import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def load_data():
  with open('dataset/data.json') as data_file:
    data = json.load(data_file)

  with open('dataset/target.json') as target_file:
    target = json.load(target_file)

  return data, target

def get_classifier(data, target):
  X_train_counts = count_vect.fit_transform(data)
  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  
  classifier = MultinomialNB().fit(X_train_tfidf, target)

  return classifier

def get_gender(twitter_message, classifier):
  X_new_counts = count_vect.transform(twitter_message)
  X_new_tfidf = tfidf_transformer.transform(X_new_counts)

  return classifier.predict(X_new_tfidf)


def main():
  data, target = load_data()
  classifier = get_classifier(data, target)

  while True:
    twitter_message = [str(input('Paste a twitter message...\n'))]

    if twitter_message[0]:
      gender = get_gender(twitter_message, classifier)
      print('gender: ', gender[0])


if __name__ == '__main__':
  main()