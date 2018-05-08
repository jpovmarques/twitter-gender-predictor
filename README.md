# twitter-gender-predictor
## Usage example
https://twitter.com/thegame/status/988913530043854851
```
tweet_message = "Short story: I was in Amsterdam & I was gifted this doll by a very sweet woman “Ellen Brudet”…"

classifier = get_classifier(data, target)
gender = get_gender(tweet_message, classifier)

print(gender)
# male

```
