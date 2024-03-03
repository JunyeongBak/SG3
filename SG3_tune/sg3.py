import platform
import subprocess
from datetime import datetime
import argparse
import os
import time

gv_datasets = os.path.join(os.getcwd(), 'input', 'Sample', 'Sample', '01.원천데이터', 'A')

# args parsing
parser = argparse.ArgumentParser(description='SG3 code')
parser.add_argument('--outdir', type=str, help='결과물 저장되는 폴더명')
parser.add_argument('--trunc', type=int, help='Truncation trick, 값이 1이면 적용안함. 1보다 작으면 다양성 감소')
parser.add_argument('--seeds', type=int, help='초기 노이즈 벡터, 0-31 형식으로 입력 시 각각 다른 시드 입력하란 의미')
parser.add_argument('--pickle', type=str, help='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/')

parser.add_argument('--source', type=str, help='원본 데이터셋 폴더')
parser.add_argument('--dest', type=str, help='리사이즈 결과물 데이터셋 zip 파일명')
parser.add_argument('--resolution', type=str, help='리사이즈 해상도')
args = parser.parse_args()


def clear_screen():
    os_name = os.name
    if os_name == 'posix':  # Unix/Linux 시스템
        os.system('clear')
    elif os_name == 'nt':  # Windows 시스템
        os.system('cls')

def inference_image(outdir:str = "out", trunc:int = 1, seeds:int = 2, pickle:str = 'stylegan3-r-afhqv2-512x512.pkl'):
    """✨추론(이미지 생성)"""
    os_name = platform.system()
    if os_name == "Windows":
        command = f'docker run --gpus all -it --rm -v %cd%:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/gen_images.py --outdir={outdir} --trunc={trunc} --seeds={seeds} --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{pickle}'
    elif os_name == "Linux":
        command = f"docker run --gpus all -it --rm --user $(id -u):$(id -g) -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/gen_images.py --outdir={outdir} --trunc={trunc} --seeds={seeds} --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{pickle}"
    else:
        raise ValueError("Unsupported OS")
    
    subprocess.run(command, shell=True)

def make_datasets(source:str = gv_datasets, dest:str = 'ffhq-256x256', resolution='256x256'):
    """✨데이터셋 리사이징"""
    os_name = platform.system()
    if os_name == 'Windows':
        command = f'docker run --gpus all -it --rm -v %cd%:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/dataset_tool.py --source={source} --dest=datasets/{dest}.zip --resolution={resolution}'
    elif os_name == 'Linux':
        command = f'docker run --gpus all -it --rm --user $(id -u):$(id -g) -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/dataset_tool.py --source={source} --dest=datasets/{dest}.zip --resolution={resolution}'
    else:
        raise ValueError('Unsupported OS')
    
    subprocess.run(command, shell=True)

def fine_tuning(outdir:str = "pickle_out"):
    """✨파인튜닝"""
    os_name = platform.system()
    if os_name == 'Windows':
        command = f'docker run --gpus all -it --rm -v %cd%:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/train.py --outdir={outdir} --cfg=stylegan3-t --data=/step1step2/input_2023090601H25M43S_256x256/datasets_256x256.zip --gpus=1 --batch=32 --batch-gpu=32 --gamma=4 --mirror=1 --kimg=5000 --snap=4 --resume=/step1step2/KHJ_20230905.pkl --cbase=16384'
    elif os_name == 'Linux':
        command = f'docker run --gpus all -it --rm --user $(id -u):$(id -g) -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/train.py --outdir=/step1step2/train_output_20230906 --cfg=stylegan3-t --data=/step1step2/input_2023090601H25M43S_256x256/datasets_256x256.zip --gpus=1 --batch=32 --batch-gpu=32 --gamma=4 --mirror=1 --kimg=5000 --snap=4 --resume=/step1step2/KHJ_20230905.pkl --cbase=16384'
    else:
        raise ValueError('Unsupported OS')
    
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
            if answer in [1,2,3,4]:
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
        print("✨데이터셋 리사이징. input폴더에 있는 데이터셋을 리사이징 합니다.")
        if args.source or args.dest or args.resolution:
            print("args를 입력하셔서 자동 생성 중...")
            lc_source = args.source if args.source else gv_datasets
            lc_dest = args.dest if args.dest else 'ffhq-256x256'
            lc_resolution = args.resolution if args.resolution else '256x256'
        else:
            print("원본 데이터셋 폴더명 입력하세요. input폴더에서 지정해야합니다.")
            try:
                lc_source = input("\033[92msource(원본 데이터셋 폴더명 입력): \033[0m").strip()
                lc_source = lc_source if lc_source else gv_datasets
                lc_dest = input("\033[92mdest(결과물 데이터셋 zip 파일명): \033[0m").strip()
                lc_dest = lc_dest if lc_dest else 'ffhq-256x256'
                lc_resolution = input("\033[92mresolution(리사이즈 해상도): \033[0m").strip()
                lc_resolution = lc_resolution if lc_resolution else '256x256'
            except Exception as e:
                print(e)

        make_datasets(source=lc_source, dest=lc_dest, resolution=lc_resolution)
    elif answer == 3:
        print("✨train(파인튜닝) 섹션")
        pass
    else:
        print("🚀프로그램을 종료합니다.")
        time.sleep(0.5)
        print("🚀Bye-Bye")
        time.sleep(1)
            
