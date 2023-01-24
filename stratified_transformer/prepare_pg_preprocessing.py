import shutil
splits = ['train', 'val']
from tqdm import tqdm

# import pdb
# pdb.set_trace()

for split in splits:
    filename = f'/cluster/51/roschmsi/ScanNet/Tasks/Benchmark/scannetv2_{split}.txt'

    with open(filename) as file:
        scenes = [line.rstrip() for line in file]

    scene_dir = f'/cluster/54/data/preprocessed/scannet/scannetv2/scans'

    filetypes = ['_vh_clean_2.ply', '_vh_clean_2.labels.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json']

    for scene_id in tqdm(scenes):
        for filetype in filetypes:
            src_path = f'{scene_dir}/{scene_id}/{scene_id}{filetype}'
            dst_path = f'/cluster/54/data/preprocessed/scannet/scannetv2/{split}/'
            shutil.copy(src_path, dst_path)
