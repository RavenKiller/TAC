import argparse
import os, sys
from multiprocessing import Process

from SensorData import SensorData

# params
# parser = argparse.ArgumentParser()
# # data paths
# parser.add_argument('--filename', required=True, help='path to sens file to read')
# parser.add_argument('--output_path', required=True, help='path to output folder')
# parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
# parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
# parser.add_argument('--export_poses', dest='export_poses', action='store_true')
# parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
# parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

# opt = parser.parse_args()

filename = ""

def extract(ids):
    print(ids)
    for id in ids:
        filename = "/root/TAC/data/rgbd_data/scannet/scans/{}/{}.sens".format(id, id)
        output_path = "/root/TAC/data/rgbd_data/scannet/val/{}".format(id)
        try:
            os.makedirs(output_path)
        except OSError:
            pass
        sd = SensorData(filename)
        sd.export_depth_images(os.path.join(output_path, 'depth'))
        sd.export_color_images(os.path.join(output_path, 'rgb'))

if __name__ == '__main__':
    with open("/root/TAC/scripts/scannet_data/scannetv2_val.txt","r") as f:
        id_list = [v.replace("\n","") for v in f.readlines()]
    M = 100
    N = 1
    id_list = id_list[:2*M]
    id_list = ["scene0660_00"]
    p_list = []
    for i in range(N):
        p = Process(target=extract, args=(id_list[i::N],))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()