import os, shutil
import subprocess
import tempfile

for direc in ["005","005b","005t","010","010b","010t","015","015b","015t","020","020b","020t","025","025b","025t","030","030b","030t","035","035b","035t","040","040b","040t","045","045b","045t","050","050b"]:
    os.chdir(direc)
    os.system("python sitest.py")
    os.chdir("..")
    print(direc+" completed.")

