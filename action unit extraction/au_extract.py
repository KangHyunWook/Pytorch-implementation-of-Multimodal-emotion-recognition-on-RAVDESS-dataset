'''
extract facial features and save action unit columns to csv.
'''

import os
import pandas as pd
import argparse

def getFilePaths(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            inner_paths = getFilePaths(full_path)
            for inner_path in inner_paths:
                pathList.append(inner_path)
    return pathList



parser = argparse.ArgumentParser()
parser.add_argument('-vid-folder', required=True)

args=parser.parse_args()

videoPaths = getFilePaths(args.vid_folder)

cols2select=['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

for videoPath in videoPaths:
    print(videoPath)
    actor_name=videoPath.split(os.path.sep)[-2]
    video_name = videoPath.split(os.path.sep)[-1].split('.')[0]
    outs=os.path.join('./outs', actor_name, video_name)

    #extract facial features from current video and save into outs folder.
    command = f'/home/jeff/codes/root/OpenFace/build/bin/FeatureExtraction -au_static -f {videoPath} -out_dir {outs}'
    os.system(command)

    #extract columns corresponding to action units from the csv file that contains all facial features.
    all_feat_file=os.path.join(outs, video_name+'.csv')
    pd_feats = pd.read_csv(all_feat_file)

    au_feats=pd_feats[cols2select]
    au_feat_save_folder=os.path.join('out_aus', actor_name)
    os.makedirs(au_feat_save_folder, exist_ok=True)
    au_feats.to_csv(os.path.join(au_feat_save_folder, video_name+'.csv'), sep=';', header=True, index=False)

