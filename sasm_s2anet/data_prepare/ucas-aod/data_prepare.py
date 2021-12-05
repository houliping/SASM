import os 
import glob
import random
import shutil

from tqdm import tqdm

random.seed(666)

def copyfiles(src_files, dst_folder, is_plane = False):
    pbar = tqdm(src_files)
    for file in pbar:
        pbar.set_description("Creating {}:".format(dst_folder))
        if not is_plane:
            filename = os.path.split(file)[1]
        else: 
            _filename = os.path.split(file)[1]
            name, ext = os.path.splitext(_filename)
            filename = 'P' + str(int(name.strip('P')) + 510).zfill(4) + ext
        dstfile = os.path.join(dst_folder, filename)
        # print(dstfile)
        shutil.copyfile(file, dstfile)


def rewrite_label(annos, dst_folder, is_plane = False):
    pbar = tqdm(annos)
    for file in pbar:
        pbar.set_description("Rewriting to {}:".format(dst_folder))
        if not is_plane:
            filename = os.path.split(file)[1]
        else: 
            _filename = os.path.split(file)[1]
            name, ext = os.path.splitext(_filename)
            filename = 'P' + str(int(name.strip('P')) + 510).zfill(4) + ext
        dstfile = os.path.join(dst_folder, filename)
        # print(dstfile)
        with open(dstfile, 'w') as fw:
            with open(file, 'r') as f:
                _lines = f.readlines()
                if is_plane:
                    lines = ['airplane  ' + x for x in _lines]  
                else:
                    lines = ['car  ' + x for x in _lines]  
                content = ''.join(lines)
                fw.write(content)

def creat_tree(root_dir):
    if not os.path.exists(root_dir):
        raise RuntimeError('invalid dataset path!')
    os.mkdir(os.path.join(root_dir, 'AllImages'))
    os.mkdir(os.path.join(root_dir, 'Annotations'))
    car_imgs = glob.glob(os.path.join(root_dir, 'CAR/*.png'))
    car_annos = glob.glob(os.path.join(root_dir, 'CAR/P*.txt'))
    airplane_imgs = glob.glob(os.path.join(root_dir, 'PLANE/*.png'))
    airplane_annos = glob.glob(os.path.join(root_dir, 'PLANE/P*.txt'))   
    copyfiles(car_imgs,  os.path.join(root_dir, 'AllImages') ) 
    copyfiles(airplane_imgs,  os.path.join(root_dir, 'AllImages'), True)
    rewrite_label(car_annos, os.path.join(root_dir, 'Annotations'))
    rewrite_label(airplane_annos, os.path.join(root_dir, 'Annotations'), True)


def generate_test(root_dir):
    setfile = os.path.join(root_dir, 'ImageSets/test.txt')
    img_dir = os.path.join(root_dir, 'AllImages')
    test_dir = os.path.join(root_dir, 'Test')
    os.makedirs(test_dir)
    if not os.path.exists(setfile):
        raise RuntimeError('{} is not founded!'.format(setfile))
    with open(setfile, 'r') as f:
        lines = f.readlines()
        pbar = tqdm(lines)
        for line in pbar:
            pbar.set_description("Copying to Test dir...")
            filename = line.strip()
            src = os.path.join(img_dir, filename + '.png')
            dst = os.path.join(test_dir, filename + '.png')
            shutil.copyfile(src, dst)



if __name__ == "__main__":
    root_dir = 'data/UCAS_AOD'
    creat_tree(root_dir)
    generate_test(root_dir)
