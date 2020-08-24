import os, glob, shutil, subprocess

def scandirs(path):
    for currentFile in glob.glob(os.path.join(path, '*') ):
        if os.path.isdir(currentFile):
            print('got a directory: ' + currentFile)
            # import ipdb; ipdb.set_trace()
            continue
            scandirs(currentFile)
        print("processing file: " + currentFile)
        tar = "tar";
        if currentFile.endswith(tar):
            # import ipdb;
            # ipdb.set_trace()
            new_folder = currentFile.split('.')[0]
            os.mkdir(new_folder)
            shutil.move(currentFile, new_folder)
            currentFile = os.path.join(new_folder, currentFile.split('/')[-1])
            subprocess.run(['tar', '-xvf', currentFile, '-C', new_folder])

path = '/data/wangzeyu/ILSRVC2012/Imagenet2012/train'
scandirs(path)