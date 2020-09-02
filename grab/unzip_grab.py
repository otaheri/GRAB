import shutil
import os
import argparse

def makepath(desired_path, isfile = False):

    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):
            os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path):
            os.makedirs(desired_path)
    return desired_path



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GRAB-unzip')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data (all zip files)')

    parser.add_argument('--extract-path', default=None, type=str,
                        help='The path to the folder to extrat GRAB to')

    args = parser.parse_args()
    zip_path =  args.grab_path
    unzip_path =  args.extract_path


    all_zips = [f for f in os.walk(zip_path)]

    if unzip_path is None:
        unzip_path = zip_path + '_unzipped'

    makepath(unzip_path)

    for dir, folder, files in all_zips:
        for file in files:

            children = file.split('__')[:-1]

            extract_dir  = os.path.join(unzip_path, *children)
            zip_name = os.path.join(dir,file)
            makepath(extract_dir)
            print(f'unzipping:\n'
                  f'{zip_name}\n'
                  f'to :\n'
                  f'{extract_dir}\n'
                  )
            shutil.unpack_archive(zip_name, extract_dir, 'zip')

    print('Unzipping finished, enjoy using GRAB dataset.')
