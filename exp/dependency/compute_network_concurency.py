import numpy as np
import cPickle as pickle

PREFIX='recognizer/recognizer_'
PREFIX_LEN=len(PREFIX)

def compute_concurency(files, out_files, threshold=0.8):
    data = []
    for fname in files:
        with open(fname, 'r') as f:
            print("Loading {}".format(fname))
            data += [pickle.load(f)]

    keys = []
    langs = []
    for report in data:
        some_key = report[0]['stats'].keys()[0]
        assert some_key.startswith(PREFIX)
        some_key = some_key[PREFIX_LEN:]
        lang = some_key[:some_key.find('/')]
        langs += [lang]

        lang_prefix_len = PREFIX_LEN + len(lang)

        keys += [set([key[lang_prefix_len:] for key in report[0]['stats'].keys()
                                            if '_initial_' not in key])]
    common_keys = keys[0]
    for s in keys[1:]:
        common_keys &= s

    means = [{} for lang in langs]

    for key in common_keys:
        print("Computing mean for {}".format(key))
        shapeFirst = data[0][0]['stats'][PREFIX+langs[0]+key].shape
        shapeLast  = data[0][-1]['stats'][PREFIX+langs[0]+key].shape
        dimsToAvg = [d for d in xrange(len(shapeFirst))
                        if shapeFirst[d] != shapeLast[d]]
        for lang_id,report in enumerate(data):
            # computing stable mean 
            mean = np.float32(0.0)
            n = np.float32(1.0)
            for entry in report:
                value = entry['stats'][PREFIX+langs[lang_id]+key]
                for dim in dimsToAvg:
                    value = value.mean(axis=dim, keepdims=True)
                mean += (value - mean)/n
                n += 1.0
            means[lang_id][key] = mean


    counts_langs = [0] * len(langs)
    mask_langs = [ {} for lang_id in xrange(len(langs)) ]
    count_all = 0
    for key in common_keys:
        for current_lang_id in xrange(len(langs)):
            mask = np.ones_like(means[current_lang_id][key], dtype=bool)
            for other_lang_id in xrange(len(langs)):
                if current_lang_id == other_lang_id:
                    continue
                curr_mask = ((means[current_lang_id][key] / means[other_lang_id][key]) < threshold)
                mask = np.logical_and(mask, curr_mask)
            mask_langs[current_lang_id][PREFIX + langs[current_lang_id] + key] = mask
            counts_langs[current_lang_id] += np.count_nonzero(mask)
        count_all += means[0][key].size

    for lang_id in xrange(len(langs)):
        print("Language {}: {}%".format(langs[lang_id], counts_langs[lang_id]*100.0/count_all))

    for lang_id, fname in enumerate(out_files):
        with open(fname, 'w') as f:
            pickle.dump(mask_langs[lang_id], f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()
    files = args.files
    assert len(files) % 2 == 0

    hlen = len(files)/2
    compute_concurency(files[:hlen], files[hlen:])

