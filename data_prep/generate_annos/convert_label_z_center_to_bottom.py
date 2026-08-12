import argparse
import glob
import os

'''
Converts SuperbAI-sourced label/*.txt from bbox-center z to bbox-bottom-
center z, which is the convention bbox3d2corners (and therefore
Avikus.__getitem__'s gt_bboxes_3d for training) expects. SuperbAI exports
always arrive as plain label/*.txt (never label_original/ or meta/*.json in
practice) with no tracklet_labels.xml (that's CVAT's signal, handled
separately by generate_cvat_label.py), so "has label/, no tracklet_labels.xml"
is what identifies a SuperbAI session here. Height is read from field 10
(0-indexed) - empirically verified against real point-cloud extents across
multiple sessions, not just assumed from the label format's field order.

Also swaps fields 8/9 (dims cols 0/1): SuperbAI's raw dims are physically
[L, W, H] (verified the same way as height - checked against real object
shape, not the field order) despite being nominally labeled W/H/L, but
anchors.py's box format needs (w, l, h). CVAT sessions get this same fix in
generate_cvat_label.py; this is the SuperbAI-side equivalent.

Each conversion runs once per session, tracked by its own marker file so
re-running annos.sh doesn't shift z / swap dims twice. They're separate
markers on purpose: sessions z-converted by an earlier version of this
script still need the dim swap applied without re-shifting z.

Label line format: name truncated occluded alpha bbox(4) dimensions(3)
location(3) rotation_y [trailing_id]. dimensions/location are fields 8:11
and 11:14 (0-indexed).
'''

EXCLUDE_DIRS = ['avikus_gt_database', 'testing', 'training', 'labels', 'meta', 'tmp']
Z_MARKER = '.z_converted'
DIMS_MARKER = '.dims_swapped'


def load_meta_dir_names(data_root):
    # Defensive only: meta/*.json has never actually shown up in real
    # SuperbAI exports, but if it ever does, don't blindly z-shift that
    # session here - keys are meta/**/*.json filenames split on the first
    # '.', not the session dir name itself.
    meta_path = os.path.join(data_root, 'meta')
    if not os.path.isdir(meta_path):
        return set()
    names = set()
    for _, _, filenames in os.walk(meta_path):
        for fname in filenames:
            if fname.lower().endswith('.json'):
                names.add(fname.split('.')[0])
    return names


def pending_conversions(data_root, dir_name, meta_dir_names):
    '''Returns the set of conversions this session still needs ('z', 'dims'),
    or an empty set if it isn't a SuperbAI session / is already fully done.'''
    session_dir = os.path.join(data_root, dir_name)
    if not os.path.isdir(os.path.join(session_dir, 'label')):
        return set()
    if os.path.isdir(os.path.join(session_dir, 'label_original')):
        return set()
    if dir_name in meta_dir_names:
        return set()
    if os.path.exists(os.path.join(session_dir, 'tracklet_labels.xml')):
        return set()

    pending = set()
    if not os.path.exists(os.path.join(session_dir, Z_MARKER)):
        pending.add('z')
    if not os.path.exists(os.path.join(session_dir, DIMS_MARKER)):
        pending.add('dims')
    return pending


def convert_session(session_dir, pending):
    label_paths = sorted(glob.glob(os.path.join(session_dir, 'label', '*.txt')))
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        new_lines = []
        for line in lines:
            fields = line.split(' ')
            if 'z' in pending:
                height = float(fields[10])
                z_center = float(fields[13])
                fields[13] = f'{z_center - height / 2:.2f}'
            if 'dims' in pending:
                # [L, W, H] -> [W, L, H]; field 10 (H) is untouched, so this is
                # order-independent w.r.t. the z shift above.
                fields[8], fields[9] = fields[9], fields[8]
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
        pending = pending_conversions(data_root, dir_name, meta_dir_names)
        if not pending:
            if os.path.isdir(os.path.join(session_dir, 'label')) and os.path.exists(os.path.join(session_dir, Z_MARKER)):
                print(f'[SKIP] {dir_name}: already converted (markers present)')
            continue

        n_converted = convert_session(session_dir, pending)
        for op, marker in (('z', Z_MARKER), ('dims', DIMS_MARKER)):
            if op in pending:
                with open(os.path.join(session_dir, marker), 'w') as f:
                    f.write('')

        applied = []
        if 'z' in pending:
            applied.append('z center -> bottom')
        if 'dims' in pending:
            applied.append('dims LWH -> WLH')
        print(f'[{dir_name}] {", ".join(applied)} ({n_converted} file(s))')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert label z from center to bottom-center for raw-capture sessions that skip generate_superb_label.py/generate_cvat_label.py')
    parser.add_argument('--data_root', type=str, required=True, help='data root containing session dirs, e.g. /workspace/data_0115')
    args = parser.parse_args()
    main(args)
