from multiprocessing import Process
import os
import wget


def download(ids):
    for id in ids:
        url = "http://kaldir.vc.in.tum.de/scannet/v1/scans/{}/{}.sens".format(id, id)
        print(url)
        outpath = "/root/TAC/data/scannet/scans/{}/{}.sens".format(id, id)
        os.makedirs("/root/TAC/data/scannet/scans/{}/".format(id), exist_ok=True)
        wget.download(url, out=outpath)


if __name__ == "__main__":
    with open("/root/TAC/scripts/scannet_data/scannetv2_val.txt", "r") as f:
        id_list = [v.replace("\n", "") for v in f.readlines()]
    M = 100
    N = 1
    id_list = id_list[:M]
    p_list = []
    for i in range(N):
        p = Process(target=download, args=(id_list[i::N],))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    # i = 0
    # download(id_list[i::N])
