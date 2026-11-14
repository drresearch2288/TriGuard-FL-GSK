import subprocess, platform, torch, urllib.request
print("python", platform.python_version(), "torch", torch.__version__)
print("cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print(subprocess.run("df -h /kaggle/working /kaggle/tmp /tmp; free -g; nproc", shell=True, capture_output=True, text=True).stdout)
print("internet", urllib.request.urlopen("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv").status)
