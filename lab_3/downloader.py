from os import listdir, rmdir, path, rename, makedirs, stat, getcwd, mkdir
from os.path import isfile, join, basename
import urllib.request
from zipfile import ZipFile
import shutil
from distutils.dir_util import copy_tree

pokemon_url = "https://github.com/kjaisingh/pokemon-classifier/archive/master.zip"
pokemon_filename = "master.zip"
pokemon_path = "data/train"
temp_dir = "temp"

class_names = ["bulbasaur", "charmander", "mewtwo", "pikachu", "squirtle"]

def download_dataset(filename, url, work_dir):
    if not path.exists(filename):
        print("[INFO] Downloading pokemon dataset....")
        filename, _ = urllib.request.urlretrieve(url, filename)
        print("nice")
        print("[INFO] Succesfully downloaded " + filename + " " + str(stat(filename).st_size) + " bytes.")
        with ZipFile(filename, 'r') as archive:
            for data in archive.namelist():
                if data.endswith('.jpg'):
                    print(work_dir)
                    archive.extract(data, work_dir)
            print("[INFO] Dataset extracted successfully.")
            archive.close()

        dst_dir = getcwd() + "/" + pokemon_path
        src_dir = getcwd() + "/temp/pokemon-classifier-master/dataset/"

        copy_tree(src_dir, dst_dir)

if __name__ == '__main__':
    if not path.exists(temp_dir):
        makedirs(temp_dir)
        makedirs(pokemon_path)

    download_dataset(pokemon_filename, pokemon_url, temp_dir)
    if path.exists(temp_dir + str('/pokemon-classifier-master/examples')):
        shutil.rmtree(temp_dir, ignore_errors=True)

