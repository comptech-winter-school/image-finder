FROM pytorch/pytorch
WORKDIR /app
COPY requirements_docker.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . .
ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
