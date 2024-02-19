FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /SG3_tune

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get upgrade -y

RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser" ]