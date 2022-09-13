FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt --no-cache-dir

RUN cp -r ./models/research/object_detection ../usr/local/lib/python3.8/site-packages/

ENTRYPOINT ["python"]

CMD ["app.py"]