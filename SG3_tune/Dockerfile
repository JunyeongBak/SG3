FROM ubuntu:22.04

WORKDIR /SG3_tune
EXPOSE 8888

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /usr/local/bin
ENV PATH="/usr/local/bin:${PATH}"
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /requirements.txt
COPY sg3_install.py /sg3_install.py
COPY sg3.py /sg3.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh


# 필수 패키지 설치
RUN apt-get update && apt-get install -y ca-certificates curl &&\
    install -m 0755 -d /etc/apt/keyrings &&\
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc &&\
    chmod a+r /etc/apt/keyrings/docker.asc 

RUN echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$UBUNTU_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

RUN chmod 755 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

