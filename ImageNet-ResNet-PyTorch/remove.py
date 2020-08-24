import os

# def scandirs(path):
#     for root, dirs, files in os.walk(path):
#         for currentFile in files:
#             print "processing file: " + currentFile
#             exts = ('.png', '.jpg')
#             if currentFile.lower().endswith(exts):
#                 os.remove(os.path.join(root, currentFile))

import os, glob

def scandirs(path):
    for currentFile in glob.glob(os.path.join(path, '*') ):
        if os.path.isdir(currentFile):
            print('got a directory: ' + currentFile)
            continue
            import ipdb; ipdb.set_trace()
            scandirs(currentFile)
        print("processing file: " + currentFile)
        jpeg = "JPEG";
        if currentFile.endswith(jpeg):
            os.remove(currentFile)

path = '/home/jumpywizard/Code/Backbone/GroupNorm-reproduce/ImageNet-ResNet-PyTorch'
scandirs(path)