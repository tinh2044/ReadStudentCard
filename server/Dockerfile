FROM python:3.9

WORKDIR /code

 COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/app

CMD ["uvicorn", "app.main:app", "--host", "127.192.0.1", "--port", "3000"]