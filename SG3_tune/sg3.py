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

# parser.add_argument('--outdir', type=str, help='결과물 저장되는 폴더명')
parser.add_argument('--cfg', type=str, help='stylegan3-t')
parser.add_argument('--data', type=str, help='datasets_256x256.zip')
parser.add_argument('--gpus', type=int, help='1')
parser.add_argument('--batch', type=int, help='32')
parser.add_argument('--batch-gpu', type=int, help='32')
parser.add_argument('--gamma', type=int, help='4')
parser.add_argument('--mirror', type=int, help='1')
parser.add_argument('--kimg', type=int, help='5000')
parser.add_argument('--snap', type=int, help='4')
parser.add_argument('--resume', type=str, help='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl')
parser.add_argument('--cbase', type=int, help='16384')

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
    os.makedirs('./datasets', exist_ok=True)
    if os_name == 'Windows':
        command = f'docker run --gpus all -it --rm -v %cd%:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/dataset_tool.py --source={source} --dest=datasets/{dest}.zip --resolution={resolution}'
    elif os_name == 'Linux':
        command = f'docker run --gpus all -it --rm --user $(id -u):$(id -g) -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/dataset_tool.py --source={source} --dest=datasets/{dest}.zip --resolution={resolution}'
    else:
        raise ValueError('Unsupported OS')
    
    subprocess.run(command, shell=True)

def fine_tuning(outdir:str = "pickle_out", cfg:str ="stylegan3-t", data:str = "datasets_256x256.zip", gpus:int = 1, batch:int = 32, batch_gpu:int = 32, gamma:int = 4, mirror:int = 1, kimg:int = 5000, snap:int = 4, resume:str = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl", cbase:int = 16384):
    """✨파인튜닝"""
    os_name = platform.system()
    if os_name == 'Windows':
        command = f'docker run --gpus all -it --rm -v %cd%:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/train.py --outdir={outdir} --cfg={cfg} --data=datasets/{data} --gpus={gpus} --batch={batch} --batch-gpu={batch_gpu} --gamma={gamma} --mirror={mirror} --kimg={kimg} --snap={snap} --resume={resume} --cbase={cbase}'
    elif os_name == 'Linux':
        command = f'docker run --gpus all -it --rm --user $(id -u):$(id -g) -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch stylegan3 python ./stylegan3/train.py --outdir={outdir} --cfg={cfg} --data=datasets/{data} --gpus={gpus} --batch={batch} --batch-gpu={batch_gpu} --gamma={gamma} --mirror={mirror} --kimg={kimg} --snap={snap} --resume={resume} --cbase={cbase}'
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
        if args.outdir or args.cfg or args.data or args.gpus or args.batch or args.batch_gpu or args.gamma or args.mirror or args.kimg or args.snap or args.resume or args.cbase:
            print("args를 입력하셔서 자동 생성 중...")
            lc_outdir = args.outdir if args.outdir else "pickle_out"
            lc_cfg = args.cfg if args.cfg else "stylegan3-t"
            lc_data = args.data if args.data else "datasets_256x256.zip"
            lc_gpus = int(args.gpus) if args.gpus else 1
            lc_batch = int(args.batch) if args.batch else 32
            lc_batch_gpu = int(args.batch_gpu) if args.batch_gpu else 32
            lc_gamma = int(args.gamma) if args.gamma else 4
            lc_mirror = int(args.mirror) if args.mirror else 1
            lc_kimg = int(args.kimg) if args.kimg else 5000
            lc_snap = int(args.snap) if args.snap else 4
            lc_resume = args.resume if args.resume else "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl"
            lc_cbase = int(args.cbase) if args.cbase else 16384
        else:
            print("파인튜닝을 시작합니다. 각 항목을 입력하세요.")
            try:
                lc_outdir = input("\033[92moutdir(결과물 저장되는 폴더명): \033[0m").strip()
                lc_outdir = lc_outdir if lc_outdir else "pickle_out"
                lc_cfg = input("\033[92mcfg(stylegan3-t): \033[0m").strip()
                lc_cfg = lc_cfg if lc_cfg else "stylegan3-t"
                lc_data = input("\033[92mdata(datasets_256x256.zip): \033[0m").strip()
                lc_data = lc_data if lc_data else "datasets_256x256.zip"
                lc_gpus = input("\033[92mgpus(1): \033[0m").strip()
                lc_gpus = int(lc_gpus) if lc_gpus else 1
                lc_batch = input("\033[92mbatch(32): \033[0m").strip()
                lc_batch = int(lc_batch) if lc_batch else 32
                lc_batch_gpu = input("\033[92mbatch_gpu(32): \033[0m").strip()
                lc_batch_gpu = int(lc_batch_gpu) if lc_batch_gpu else 32
                lc_gamma = input("\033[92mgamma(4): \033[0m").strip()
                lc_gamma = int(lc_gamma) if lc_gamma else 4
                lc_mirror = input("\033[92mmirror(1): \033[0m").strip()
                lc_mirror = int(lc_mirror) if lc_mirror else 1
                lc_kimg = input("\033[92mkimg(5000): \033[0m").strip()
                lc_kimg = int(lc_kimg) if lc_kimg else 5000
                lc_snap = input("\033[92msnap(4): \033[0m").strip()
                lc_snap = int(lc_snap) if lc_snap else 4
                lc_resume = input("\033[92mresume: \033[0m").strip()
                lc_resume = lc_resume if lc_resume else "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl"
                lc_cbase = input("\033[92mcbase(16384): \033[0m").strip()
                lc_cbase = int(lc_cbase) if lc_cbase else 16384
            except Exception as e:
                print(e)
        fine_tuning(outdir=lc_outdir, cfg=lc_cfg, data=lc_data, gpus=lc_gpus, batch=lc_batch, batch_gpu=lc_batch_gpu, gamma=lc_gamma, mirror=lc_mirror, kimg=lc_kimg, snap=lc_snap, resume=lc_resume, cbase=lc_cbase)
    else:
        print("🚀프로그램을 종료합니다.")
        time.sleep(0.5)
        print("🚀Bye-Bye")
        time.sleep(1)
