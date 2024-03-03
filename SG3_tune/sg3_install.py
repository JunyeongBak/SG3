import subprocess
import platform

def pyversion():
    os_name = platform.system()
    if os_name == "Windows":
        command = 'python --version'
    elif os_name == "Linux":
        command = 'python3 --version'
    else:
        raise ValueError("Unsupported OS")
    subprocess.run(command, shell=True)

def install():
    command = 'git clone https://github.com/NVlabs/stylegan3.git'
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    print("\033[92m📢2024논문구현 1기 SG3 환경구축\033[0m")
    pyversion()
    install()