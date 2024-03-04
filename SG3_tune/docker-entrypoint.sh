#!/bin/bash
exec \"$@\"

echo "Docker 데몬을 시작합니다."
systemctl --user start docker-desktop
docker --version

echo "Python 스크립트 sg3_install.py를 실행합니다."
python3 sg3_install.py

echo "⬇⬇⬇아래 명령어를 복사해서 새로운 터미널 창에서 실행하세요."
echo "경로는 SG3/SG3_tune/stylegan3"
echo "docker build --tag stylegan3 ."

# 스크립트가 종료되지 않고 계속 실행되도록 대기 상태 유지
# 이는 컨테이너가 바로 종료되는 것을 방지합니다.
tail -f /dev/null
