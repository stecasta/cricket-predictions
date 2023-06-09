FROM python:3.9

WORKDIR /app

ADD . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "infer.py", "--test"]