import requests
from bs4 import BeautifulSoup
import re
import os
PREFIX = "https://vision.in.tum.de/"
res = requests.get("https://vision.in.tum.de/data/datasets/rgbd-dataset/download")
soup = BeautifulSoup(res.text)
linktags = soup.find_all('a', href=re.compile("/rgbd/dataset.*?tgz"))
linktags = linktags[:(len(linktags)//2)]
links = []
for a in linktags:
    relative = a.get("href")
    if "checkerboard_large" not in relative and "calibration" not in relative:
        links.append(PREFIX+relative)
FOLDER = "/home/zqy/hzt/UniSpeaker/data/rgbd_data/tumrgbd/"
for link in links:
    cmd = "wget -P %s %s"%(FOLDER, link)
    os.system(cmd)   