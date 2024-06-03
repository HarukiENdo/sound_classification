import os
import glob
import shutil
import random
random.seed(888)
import numpy as np
import soundfile as sf
import torchaudio

data_dir=""
print('実行前')
dir_in = '/car'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/cut'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/environ'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/fruit'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/leaf'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/truck'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/unknown'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/walk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/talk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))

#このデータセットは外部で拡張されているので、ここで検証データ用に分けておく valid,testは各クラスの10%ずつ
data_dir=""
val_dir=""
test_dir = ""
#--------------------------------------------------------------------------------------------------carの検証データ用意
mv_file=255
dir_in =os.path.join(data_dir+'/car')
dir_val=os.path.join(val_dir+'/car')
dir_test = os.path.join(test_dir+'/car')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name) 
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------cutの検証データ用意
mv_file=139
dir_in =os.path.join(data_dir+'/cut')
dir_val=os.path.join(val_dir+'/cut')
dir_test = os.path.join(test_dir+'/cut')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue
    
#-------------------------------------------------------------------------------------------------environの検証データ用意
mv_file=37847
dir_in =os.path.join(data_dir+'/environ')
dir_val=os.path.join(val_dir+'/environ')
dir_test = os.path.join(test_dir+'/environ')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------fruitの検証データ用意
mv_file=372
dir_in =os.path.join(data_dir+'/fruit')
dir_val=os.path.join(val_dir+'/fruit')
dir_test = os.path.join(test_dir+'/fruit')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------leafの検証データ用意
mv_file=396
dir_in =os.path.join(data_dir+'/leaf')
dir_val=os.path.join(val_dir+'/leaf')
dir_test = os.path.join(test_dir+'/leaf')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    shutil.move(file_path,val_file_path)
    print(f"moved{file_name} to {dir_val}")

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------talkの検証データ用意
mv_file=1935
dir_in =os.path.join(data_dir+'/talk')
dir_val=os.path.join(val_dir+'/talk')
dir_test = os.path.join(test_dir+'/talk')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------truckの検証データ用意
mv_file=11
dir_in =os.path.join(data_dir+'/truck')
dir_val=os.path.join(val_dir+'/truck')
dir_test = os.path.join(test_dir+'/truck')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------unknownの検証データ用意
mv_file=320
dir_in =os.path.join(data_dir+'/unknown')
dir_val=os.path.join(val_dir+'/unknown')
dir_test = os.path.join(test_dir+'/unknown')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

#-------------------------------------------------------------------------------------------------walkの検証データ用意
mv_file=9051
dir_in =os.path.join(data_dir+'/walk')
dir_val=os.path.join(val_dir+'/walk')
dir_test = os.path.join(test_dir+'/walk')
filepaths=glob.glob(dir_in+'/*')
random_files_val=random.sample(filepaths,mv_file)
for file_path in random_files_val:
    file_name=os.path.basename(file_path)
    val_file_path=os.path.join(dir_val,file_name)           
    try:
        shutil.move(file_path,val_file_path)
        print(f"moved{file_name} to {dir_val}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

random_files_test=random.sample(filepaths,mv_file)
for file_path in random_files_test:
    file_name=os.path.basename(file_path)
    test_file_path=os.path.join(dir_test,file_name)           
    try:
        shutil.move(file_path,test_file_path)
        print(f"moved{file_name} to {dir_test}")
    except FileNotFoundError:
        print(f"File{file_name} not found. Skipping...")
        continue

print('終了')

data_dir=""
print('実行後')
dir_in = '/car'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/cut'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/environ'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/fruit'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/leaf'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/truck'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/unknown'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/walk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/talk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))

data_dir=""
print('実行後')
dir_in = '/car'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/cut'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/environ'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/fruit'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/leaf'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/truck'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/unknown'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/walk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/talk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))

data_dir=""
print('実行後')
dir_in = '/car'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/cut'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/environ'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/fruit'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/leaf'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/truck'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/unknown'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/walk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/talk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))