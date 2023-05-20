FROM python:3.9
COPY . /app
WORKDIR /app
# optional: RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 80
WORKDIR /app
CMD ["python3", "-m", "streamlit", "run", "app3.py"]