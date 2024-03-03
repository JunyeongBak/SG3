import platform
import subprocess
from datetime import datetime
import argparse
import os
import time

# args parsing
parser = argparse.ArgumentParser(description='SG3 code')
parser.add_argument('--outdir', type=str, help='결과물 저장되는 폴더명')
parser.add_argument('--trunc', type=int, help='Truncation trick, 값이 1이면 적용안함. 1보다 작으면 다양성 감소')
parser.add_argument('--seeds', type=int, help='초기 노이즈 벡터, 0-31 형식으로 입력 시 각각 다른 시드 입력하란 의미')
parser.add_argument('--pickle', type=str, help='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/')
args = parser.parse_args()


def clear_screen():
    os_name = os.name
    if os_name == 'posix':  # Unix/Linux 시스템
        os.system('clear')
    elif os_name == 'nt':  # Windows 시스템
        os.system('cls')

def inference_image(outdir:str = "out", trunc:int = 1, seeds:int = 2, pickle:str = 'stylegan3-r-afhqv2-512x512.pkl', **kwargs):
    os_name = platform.system()
    if os_name == "Windows":
        command = f'docker run --gpus all -it --rm -v %cd%:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/gen_images.py --outdir={outdir} --trunc={trunc} --seeds={seeds} --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{pickle}'
    elif os_name == "Linux":
        command = f"docker run --gpus all -it --rm --user $(id -u):$(id -g) -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/gen_images.py --outdir={outdir} --trunc={trunc} --seeds={seeds} --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{pickle}"
    else:
        raise ValueError("Unsupported OS")
    
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    while True:
        clear_screen()
        print("\033[93m📢어떤 작업을 원하시나요?\n① inference(생성)\n② train(파인튜닝)\n③ 종료\033[0m")
        try:
            answer = input("\033[92m정수만 입력하세요: \033[0m").strip()
        except Exception as e:
            print(e)
        
        if answer and answer.isdigit():
            answer = int(answer)
            if answer in [1,2,3]:
                break
        print("잘못된 입력. \033[91m정수1,2,3\033[0m만 입력 가능!")
        print("🚀2초 후 초기화면으로 갑니다.")
        time.sleep(1)
        print("🚀1초 후 초기화면으로 갑니다.")
        time.sleep(1)
    
    if answer == 1:
        print("✨Inference(생성 섹션)")
        if args.outdir or args.seeds or args.trunc or args.pickle:
            print("args를 입력하셔서 자동 생성 중...")
            lc_outdir = args.outdir if args.outdir else 'output'
            lc_seeds = int(args.seeds) if args.seeds else 2
            lc_trunc = int(args.trunc) if args.trunc else 1
            lc_pickle = args.pickle if args.pickle else 'stylegan3-r-afhqv2-512x512.pkl'
        else:
            print("결과물 저장되는 폴더명 입력하세요. Format 폴더명_YYYY-MM-DD_HH_MM_SS")
            try:
                lc_outdir = input("\033[92moutdir(폴더명 입력): \033[0m").strip()
                lc_outdir = lc_outdir if lc_outdir else "output"
                lc_seeds = input("\033[92mseeds(초기 노이즈 벡터 입력): \033[0m").strip()
                lc_seeds = int(lc_seeds) if lc_seeds else 2
                lc_trunc = input("\033[92mtrunc(Truncation trick입력): \033[0m").strip()
                lc_trunc = int(lc_trunc) if lc_trunc else 1
                lc_pickle = input("\033[92mPretrainedpickle(https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/)입력: \033[0m").strip()
                lc_pickle = lc_pickle if lc_pickle else "stylegan3-r-afhqv2-512x512.pkl"
            except Exception as e:
                print(e)
        lc_outdir = f"{lc_outdir}_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
        inference_image(outdir=lc_outdir, trunc = lc_trunc, seeds=lc_seeds, pickle=lc_pickle)
    elif answer == 2:
        print("✨Train (파인튜닝 섹션)")
        time.sleep(1)
    else:
        print("🚀프로그램을 종료합니다.")
        time.sleep(0.5)
        print("🚀Bye-Bye")
        time.sleep(1)
            
