'''
code by zzg@2021/05/10
'''
import os.path as osp
import os
import numpy as np
import shutil
import re
import sys

FLAG = 1
if FLAG == 1:

    ##train
    src_dir = "D:\yolo\yolov7-main\data1\Image\DETRAC-train-data" #image
    dst_dir = "D:\yolo\yolov7-main\sjcl\sjcl/train_image"

else:
    ##test
    src_dir = "D:\yolo\yolov7-main\data1\Image\DETRAC-test-data" #image
    dst_dir = "D:\yolo\yolov7-main\sjcl\sjcl/test_image"

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

mkdirs(dst_dir)

seqs = [s for s in os.listdir(src_dir)]

for seq in seqs: 
    path = osp.join(src_dir, seq)
    # print(path)
    fileList = os.listdir(path)  
    os.chdir(path)  

    for fileName in fileList: 
        pat = ".+\.(jpg|jpeg|JPG)"  
        pattern = re.findall(pat, fileName)  
        image_name = fileName.split(".")[0]  

        os.rename(fileName, ('{}_{}.jpg'.format(image_name, str(seq)))) 
       
    sys.stdin.flush()  
    print("after rename：" + str(os.listdir(path))) 

print("step1 finished!-----")

j = 0
for seq in seqs: 
    path = osp.join(src_dir, seq)

    for root, dirs, files in os.walk(path):
        files = sorted(files)
        for i in range(len(files)):
            if i%10== 0: 
                j += 1
                if (files[i][-3:] == 'jpg' or files[i][-3:] == 'JPG') or files[i][-4:]=='jpeg':

                    file_path = path + '/' + files[i]
                    new_file_path = dst_dir + '/' + files[i]
                    shutil.copy(file_path, new_file_path)
                print(j)
print("step2 finished!------")
# print(str(os.listdir(dst_dir)))