FROM python
RUN apt-get update -y && apt-get install -y build-essential
ADD ./python /code
WORKDIR /code
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]