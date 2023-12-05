FROM --platform=linux/amd64 tensorflow/tensorflow:2.13.0

COPY requirements.txt /requirements.txt
RUN pip install -U pip
RUN pip install -r /requirements.txt

COPY quizachu /quizachu

USER 0
RUN mkdir -p /home/rob/.cache/models/score_model
USER $CONTAINER_USER_ID

CMD uvicorn quizachu.api.fast:app --host 0.0.0.0 --port $PORT
