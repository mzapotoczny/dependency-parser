from __future__ import print_function
import glob
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(usage='convert_data_params.py prefix lang_main lang2 lang3 (folder names)...')
    parser.add_argument('--all-in-one', action='store_true')
    parser.add_argument('--no-test', action='store_true')
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()
    allinone = args.all_in_one
    files = args.files
    notest = args.no_test

    assert len(files) >= 2, str(files)
    prefix = files[0]
    lang_names = files[1:]

    params = []
    all_splits = ['train', 'dev']
    if not notest:
        all_splits += ['test']

    for lang in lang_names:
        path = os.path.join(prefix, lang)
        for split in all_splits:
            glob_path = path+'/*-'+split+'.conllu'
            files = glob.glob(glob_path)
           
            assert len(files) == 1, (str(files), glob_path)
            data = files[0]
            split_name_suffix = '_'+data[len(path)+1:][:2]
            if lang == lang_names[0] or allinone:
                split_name_suffix = ''
            params += [ '{}{}={}'.format(split, split_name_suffix, data) ]
    print(' '.join(params), end='')

