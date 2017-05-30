#!/usr/bin/env python
from __future__ import print_function
from contextlib import closing
import zipfile
import sys
import os.path

def compact(finname, foutname):
    with closing(zipfile.ZipFile(finname, 'r')) as zin:
        with closing(zipfile.ZipFile(foutname, 'w')) as zout:
            for item in zin.infolist():
                fname = item.filename
                if fname.startswith('_') or fname.startswith('recognizer'):
                    buff = zin.read(item)
                    zout.writestr(item, buff)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--same-folder', action='store_true',
        help='If set we treat FILES argument as input files and store the result\
        in the same folder. Otherwise we treat FILES as input1 output1 input2\
        output2...')

    args = parser.parse_args()
    files = args.files
    same_folder = args.same_folder

    if not same_folder and len(files) % 2 == 1:
        print("ERROR: Without --same-folder number of FILES must be even.",
                file=sys.stderr)
        sys.exit(1)

    fnames = []

    def check_file(finname):
        if not os.path.isfile(finname):
            print("ERROR: File {} does not exists. Nothing done.".format(finname),
                    file=sys.stderr)
            sys.exit(1)

    if same_folder:
        for fname in files:
            check_file(fname)
            fnames += [(fname, fname[:-4]+"_compact.zip")]
    else:
        for finname, foutname in zip(files[::2], files[1::2]):
            check_file(finname)
            fnames += [(finname, foutname)]

    for finname, foutname in fnames:
        print("Compacting {} to {}.".format(finname, foutname))
        compact(finname, foutname)

