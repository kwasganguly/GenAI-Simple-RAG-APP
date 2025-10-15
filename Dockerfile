FROM python:3.10.0
EXPOSE 8080
WORKDIR /app
COPY . ./
RUN pip3 install --upgrade pip
RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate
RUN pip3 install -r requirements.txt
ENTRYPOINT ["streamlit" , "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]