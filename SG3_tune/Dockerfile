FROM ubuntu:22.04

WORKDIR /SG3_tune
EXPOSE 8888

COPY requirements.txt requirements.txt
COPY sg3_install.py /SG3_tune/sg3_install.py
COPY sg3.py /SG3_tune/sg3.py

RUN apt-get update && \
    apt-get install -y git python3.9 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["python3", "sg3_install.py"]
