FROM ubuntu
WORKDIR /usr/src/app
RUN rm -rf /var/lib/apt/list/* && apt-get update && apt-get install python3 python3-pip -y
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install tiktoken
RUN pip3 install gunicorn

ADD . .

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
