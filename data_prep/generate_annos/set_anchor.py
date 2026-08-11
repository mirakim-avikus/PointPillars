import os
import pickle
import argparse
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE, '../../')))

from pointpillars.dataset import Avikus

MODEL_FILE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pointpillars', 'model', 'pointpillars.py'))

# Row order of pointpillars.py's `sizes` anchor list follows Avikus.CLASSES'
# index order (jetski, smallboat, mediumboat, c-marker, yacht, pole,
# dinghyboat, bigboat, bridgepillar, buoy).
CLASS_ORDER = sorted(Avikus.CLASSES, key=Avikus.CLASSES.get)


def update_model_anchor_sizes(wlh_by_class):
    '''Rewrite the rows of pointpillars.py's `sizes = [[...]]` anchor list
    for classes present in wlh_by_class, leaving other rows untouched.
    Returns the 1-indexed line number of the `sizes` assignment, or None if
    the block couldn't be found/patched.'''
    with open(MODEL_FILE, 'r') as f:
        text = f.read()

    block_match = re.search(r'sizes\s*=\s*\[\[.*?\]\]', text, re.DOTALL)
    if block_match is None:
        print(f'[set_anchor.py] WARNING: could not find `sizes = [[...]]` in {MODEL_FILE}, skip auto-update')
        return None

    block = block_match.group(0)
    row_matches = list(re.finditer(r'\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]', block))
    if len(row_matches) != len(CLASS_ORDER):
        print(f'[set_anchor.py] WARNING: expected {len(CLASS_ORDER)} anchor rows in {MODEL_FILE}, found {len(row_matches)}, skip auto-update')
        return None

    new_block = block
    for idx in range(len(row_matches) - 1, -1, -1):
        cls = CLASS_ORDER[idx]
        if cls not in wlh_by_class:
            continue
        w, l, h = wlh_by_class[cls]
        m = row_matches[idx]
        new_block = new_block[:m.start()] + f'[{w:.2f}, {l:.2f}, {h:.2f}]' + new_block[m.end():]

    text = text[:block_match.start()] + new_block + text[block_match.end():]
    with open(MODEL_FILE, 'w') as f:
        f.write(text)

    line_no = text[:block_match.start()].count('\n') + 1
    return line_no


def main(args):
    args.data_root = os.path.normpath(args.data_root)
    root_path = args.data_root
    with open(os.path.join(root_path, "avikus_dbinfos_train.pkl"), 'rb') as f :
        data = pickle.load(f)

    classes = data.keys()
    print("======= PointPillars anchors per class (used for training) =======")
    print("W L H")
    wlh_by_class = {}
    for cls in classes:
        # box3d_lidar[3:6] is written by generate_cvat_label.py's `dimensions`
        # field, already in anchors.py's (w, l, h) order - no reordering needed
        # here.
        wlh = [0, 0, 0]
        for dct in data[cls]:
            wlh += dct['box3d_lidar'][3:6]
        wlh /= len(data[cls])

        print(f'{cls} : {round(wlh[0], 2)}, {round(wlh[1], 2)}, {round(wlh[2], 2)}')
        wlh_by_class[cls] = (wlh[0], wlh[1], wlh[2])

    line_no = update_model_anchor_sizes(wlh_by_class)
    if line_no is not None:
        print(f'these values are automatically sent to the pointpillars anchor size at pointpillars/model/pointpillars.py line {line_no}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="anchor setting script")
    parser.add_argument("--data_root", type=str, default="data", help="Path to the data folder")
    args = parser.parse_args()

    main(args)
