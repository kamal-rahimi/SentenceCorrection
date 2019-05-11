# Sentence words order correction using Recurrent Neural Networks

## Model description
The model consists of an Embedding Layer followed by a uni-directional Recurrent Neural networks and a fully connected layer.

The network stucture is depicted below: 


							Predicted next word

						       0 1 2 3 . . . . . 9999		   
						       | | | | . . . . .   | 
						      |---------------------|
						      | 	softmax     |
						      |---------------------|
								|
						     |-----------------------|
						     | fully-connected (512) |
						     | ACT: ELU              |
						     |-----------------------|
								|
		       |-----------------------|		|
		       |       LSTM (70)       |----------------|
		       |-----------------------|
			 	  |
			|---------------------|
			|Embedding (dim=40)   |
			|Input size=10000     |
			|---------------------|			
			 | | | | . . . . .   |
			 0 1 2 3 . . . . . 9999

		       Input sentence word sequnce


The model is trained to predict the next word in a sentence, given previous words of the sentence. Cross enthropy is used to measure model performance with 'accuracy' as the metric.

Once the model is trained, the condidtional probability of each word of a sentence, given its predecessor words, is estimated by the model. The sentence liklihood is calculated based condidtional probabilities of its words. For instance, the probabity of seneynces compoised of words $w_1, w_2, ...w_n$ is calculates as:

$P(w_1, w_2, ..., w_n) = P(w_1) x P(w_2|w_1) x ... x P(w_n | w_{n-1}, ...w_1)$

where $P(w_i | w_{i-1}, ...w_1)$ is the output conditional probabity estimated by the network for i'th word given w_1 to w_{i-1}.

To correct word order in a given input sentence, the model compares liklihood of sentences constcructed by a window-based permutations of words and outputs a sentence corresponding to a word order with the highest liklihood in the longuage model. The model is also used to finish an input sequence of words as a sentence
by predicting next sentence words.



## How to use the model
 1. Directly running the scripts
 2. Using the web UI (developed using flask)
 3. Deploying the model in container (based on Docker) and using web UI

### train_lm.py
Builds a longuage model to estimate the liklihood of any sentence

Example usage:
```
$ python3 train_lm.py -n "English" -p "/data/english.txt"
```
Note a model with name "English" has already been trained in this repository using 16000 English sentences.


### process_sentence.py
Estimates the liklihood of an input senetnce and finds an order of words
that is most likely in the longuage model

Example usage:
```
$ python3 process_sentence.py -n "English" -s "cat is there in a the room"
Using TensorFlow backend.

Input:
cat is there in a the room
Liklihood:
4.879379846492504e-33

Corrected:
there is a cat in the room
Liklihood:
3.088558734790202e-14

$ python3 process_sentence.py -n "English" -s "rain it tomorrow  will"

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


### finish_sentence.py
Completes an input sequnce of words  as a sentnces (finish sentence)

Example usage:
```
$ python3 finish_sentence.py -n "English" -s "Life is too short to " -c 5

Using TensorFlow backend.

<BGN> life is too short to insist <EOS>
<BGN> life is too short to worry of running that returned with you
<BGN> life is too short to win at the bottom <EOS>
<BGN> life is too short to gather under ourselves management to force through
<BGN> life is too short to commence to the public <EOS>

$ python3 finish_sentence.py -n "English" -s "many people think" -c 5

Using TensorFlow backend.

<BGN> many people think of buying the other hand while prepared under their
<BGN> many people think all the blinds is a cool and wait bill
<BGN> many people think the larger regular enforcement were melted down by a
<BGN> many people think all the numbers was arrested by a less irritating
<BGN> many people think that he would go anywhere said he <EOS>
```

### run_server.py
Creates a web inteface to use the model for word order correction and sentence creation (using a model trained for English).

Example usage:
```
$ python3 run_server.py
Using TensorFlow backend.
 * Serving Flask app "run_server" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:8911/ (Press CTRL+C to quit)
```

The web interface can be accessed in a browser when navigating to http://0.0.0.0:8911

### Dockerfile
Creates a container using Docker and creates a web interface to use the model (all required packages are installed
automatically in the Docker container).

Example usage:
```
$ sudo docker build -t wordlm .
$ sudo docker run -p 8911:8911 -it  wordlm python3 run_server.py
```

Navigate to http://0.0.0.0:8911 in a broswer to accees the model web interface.
