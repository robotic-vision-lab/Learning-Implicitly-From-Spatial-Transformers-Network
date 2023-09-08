import os
from glob import glob

fls = glob('./data/DISN_split/*_test.lst')
print(len(fls))

for sfl in fls:
    c = sfl.split('/')[-1].split('_')[0]
    files = open(sfl).readlines()
    print(c, len(files))
    for f in files:
        if f.strip() == 'd93133f1f5be7da1191c3762b497eca9':
            print(c, f, 'found')
        if not os.path.exists('/work/06035/sami/maverick2/datasets/shapenet/disn/image/'+c+'/'+f.strip()):
            print(c, f)


