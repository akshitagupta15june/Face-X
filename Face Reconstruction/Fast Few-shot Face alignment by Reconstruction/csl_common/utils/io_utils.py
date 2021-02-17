import os


def makedirs(path):
    out_dir = path
    if os.path.splitext(path)[1]:  # file
        out_dir = os.path.split(path)[0]
    if out_dir:
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass

def copy_files(src_dir, dst_dir, pattern):
    """
    Copy all files pattern.
    """
    import glob
    import shutil
    for src_file in glob.glob(os.path.join(src_dir, )):
        dir_, fname = os.path.split(src_file)
        # dst_file = os.path.join(os.path.dirname(dir_), fname)
        dst_file = os.path.join(dst_dir, fname)
        shutil.copyfile(src=src_file, dst=dst_file)