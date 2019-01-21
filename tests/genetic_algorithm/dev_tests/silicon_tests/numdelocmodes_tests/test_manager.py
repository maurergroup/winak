import os, shutil
import subprocess
import tempfile

for direc in ["025","050","075"]:
    os.chdir(direc)
    os.system("python sitest.py")
    os.chdir("..")
    print(direc+" completed.")

