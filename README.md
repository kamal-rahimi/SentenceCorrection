# WordLM
Sentence words order correction (using a language model)

## train_lm.py
Builds a longuage model to estimate the liklihood of any sentence

Example usage:
```
$ python train_lm.py -n "English" -p "/data/english.txt"

```
Note a model with name "English" has already been trained in this repository using 16000 English sentences.



## process_sentence.py
Estimates the liklihood of an input senetnce and finds an order of words
that is most likely in the longuage model

Example usage:
```
$ python process_sentence.py -n "English" -s "cat is there in a the room"
Using TensorFlow backend.

Input: 
cat is there in a the room
Liklihood:
4.879379846492504e-33

Corrected: 
there is a cat in the room
Liklihood:
3.088558734790202e-14


$ python process_sentence.py -n "English" -s "rain it tomorrow  will"

Using TensorFlow backend.

Input: 
rain it tomorrow  will
Liklihood:
5.288667027438841e-24

Corrected: 
tomorrow it will rain
Liklihood:
1.202084873758493e-12

```

## finish_sentence.py
Completes an input sequnce of words  as a sentnces (finish sentence)

Example usage:
```
$ python finish_sentence.py -n "English" -s "Life is too short to " -c 5

Using TensorFlow backend.

<BGN> life is too short to insist <EOS>
<BGN> life is too short to worry of running that returned with you
<BGN> life is too short to win at the bottom <EOS>
<BGN> life is too short to gather under ourselves management to force through
<BGN> life is too short to commence to the public <EOS>

$ python finish_sentence.py -n "English" -s "many people think" -c 5

Using TensorFlow backend.

<BGN> many people think of buying the other hand while prepared under their
<BGN> many people think all the blinds is a cool and wait bill
<BGN> many people think the larger regular enforcement were melted down by a
<BGN> many people think all the numbers was arrested by a less irritating
<BGN> many people think that he would go anywhere said he <EOS>

```

