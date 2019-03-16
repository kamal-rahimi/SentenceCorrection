FROM ubuntu:18.04

RUN apt-get update --fix-missing && apt-get install -y python3 python3-pip

ADD . /home/word_lm

WORKDIR /home/word_lm

RUN pip3 install -r requirements.txt

RUN apt-get clean

CMD ["/bin/bash"]
