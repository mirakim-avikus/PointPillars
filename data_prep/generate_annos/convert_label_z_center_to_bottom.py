import argparse
import glob
import os

'''
generate_superb_label.py and generate_cvat_label.py both convert their raw
label sources' center-z to bottom-center-z, which is the convention
bbox3d2corners (and therefore Avikus.__getitem__'s gt_bboxes_3d for
training) expects. Raw captures that arrive with label/*.txt already
populated skip both of those scripts entirely (no label_original/,
meta/*.json, or tracklet_labels.xml to trigger them), so their z is still
center-z. This finds those sessions and does the same center -> bottom
conversion, once per session (tracked via a marker file so re-running
annos.sh doesn't shift z twice).

Label line format: name truncated occluded alpha bbox(4) dimensions(3)
location(3) rotation_y [trailing_id]. dimensions/location are fields 8:11
and 11:14 (0-indexed).
'''

EXCLUDE_DIRS = ['avikus_gt_database', 'testing', 'training', 'labels', 'meta', 'tmp']
MARKER_NAME = '.z_converted'


def load_meta_dir_names(data_root):
    # Mirrors generate_superb_label.py's meta_by_dir: keys are meta/**/*.json
    # filenames split on the first '.', not the session dir name itself.
    meta_path = os.path.join(data_root, 'meta')
    if not os.path.isdir(meta_path):
        return set()
    names = set()
    for _, _, filenames in os.walk(meta_path):
        for fname in filenames:
            if fname.lower().endswith('.json'):
                names.add(fname.split('.')[0])
    return names


def needs_conversion(data_root, dir_name, meta_dir_names):
    session_dir = os.path.join(data_root, dir_name)
    if not os.path.isdir(os.path.join(session_dir, 'label')):
        return False
    if os.path.isdir(os.path.join(session_dir, 'label_original')):
        return False
    if dir_name in meta_dir_names:
        return False
    if os.path.exists(os.path.join(session_dir, 'tracklet_labels.xml')):
        return False
    if os.path.exists(os.path.join(session_dir, MARKER_NAME)):
        return False
    return True


def convert_session(session_dir):
    label_paths = sorted(glob.glob(os.path.join(session_dir, 'label', '*.txt')))
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        new_lines = []
        for line in lines:
            fields = line.split(' ')
            height = float(fields[10])
            z_center = float(fields[13])
            fields[13] = f'{z_center - height / 2:.2f}'
            new_lines.append(' '.join(fields))

        with open(label_path, 'w') as f:
            f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))

    return len(label_paths)


def main(args):
    data_root = os.path.normpath(args.data_root)
    meta_dir_names = load_meta_dir_names(data_root)

    for dir_name in sorted(os.listdir(data_root)):
        session_dir = os.path.join(data_root, dir_name)
        if dir_name in EXCLUDE_DIRS or not os.path.isdir(session_dir):
            continue
        if not needs_conversion(data_root, dir_name, meta_dir_names):
            if os.path.exists(os.path.join(session_dir, 'label')) and os.path.exists(os.path.join(session_dir, MARKER_NAME)):
                print(f'[SKIP] {dir_name}: already z-converted (marker present)')
            continue

        n_converted = convert_session(session_dir)
        with open(os.path.join(session_dir, MARKER_NAME), 'w') as f:
            f.write('')
        print(f'[{dir_name}] label z center -> bottom converted ({n_converted} file(s))')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert label z from center to bottom-center for raw-capture sessions that skip generate_superb_label.py/generate_cvat_label.py')
    parser.add_argument('--data_root', type=str, required=True, help='data root containing session dirs, e.g. /workspace/data_0115')
    args = parser.parse_args()
    main(args)
