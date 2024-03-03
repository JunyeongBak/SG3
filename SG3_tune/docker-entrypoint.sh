#!/bin/sh
# Docker 데몬 시작
# DinD 환경에서는 일반적으로 필요하지 않을 수 있으나, 
# 필요한 초기화 작업이 있다면 여기에 추가합니다.

# 예를 들어, Docker 이미지를 빌드하고 실행하는 명령을 추가할 수 있습니다.
# docker build -t my-app .
# docker run my-app
echo "Docker 데몬을 시작합니다."
systemctl --user start docker-desktop
docker --version


echo "Python 스크립트 sg3_install.py를 실행합니다."
python3 sg3_install.py

docker build --tag stylegan3 .
# 스크립트가 종료되지 않고 계속 실행되도록 대기 상태 유지
# 이는 컨테이너가 바로 종료되는 것을 방지합니다.
tail -f /dev/null