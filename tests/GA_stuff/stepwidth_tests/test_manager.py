import os, shutil
import subprocess
import tempfile

for direc in ["100_999","100_999t","125_999","125_999b","125_999t","150_999","150_999b","150_999t","175_999","175_999b","175_999t","200_999","200_999b","200_999t","225_999","225_999b","225_999t","250_999","250_999b","250_999t","275_999","275_999b","275_999t","300_999","300_999b","300_999t"]:
    os.chdir(direc)
    os.system("python sitest.py")
    os.chdir("..")
    print(direc+" completed.")

