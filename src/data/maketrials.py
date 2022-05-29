from gc import collect
import os
import glob
import random
import itertools

random.seed(1234)


def basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def traverse(root, split='dev'):
    spk_to_files = {}

    for spk in os.listdir(os.path.join(root, split + '-clean')):
        spk_to_files[spk] = glob.glob(os.path.join(root, split + '-clean', spk, '*', '*.flac'))

    for spk in os.listdir(os.path.join(root, split + '-other')):
        spk_to_files[spk] = glob.glob(os.path.join(root, split + '-other', spk, '*', '*.flac'))

    return spk_to_files


def make_trials(spk_to_files, num=100):
    positive_pairs = {}
    negative_pairs = {}

    for k in spk_to_files.keys():
        files = spk_to_files[k]

        # choise positive pairs
        combs = itertools.combinations(files, 2)
        positive_pairs[k] = random.sample(list(combs), num)

        # choise negative pairs
        negatives = [f for neg in spk_to_files.keys() if neg != k for f in spk_to_files[neg]]
        combs = itertools.product(files, negatives)
        negative_pairs[k] = random.sample(list(combs), num)

    return positive_pairs, negative_pairs


def write_trials(positive_pairs, negative_pairs, output):
    with open(output, 'w') as f:
        for k in positive_pairs.keys():
            for enrollment, test in positive_pairs[k]:
                f.write(f'1 {basename(enrollment)} {basename(test)}\n')

        for k in negative_pairs.keys():
            for enrollment, test in negative_pairs[k]:
                f.write(f'0 {basename(enrollment)} {basename(test)}\n')


def main():
    root = 'data/materials/LibriSpeech/'
    dev_trials = 'data/materials/dev-trials.txt'
    test_trials = 'data/materials/test-trials.txt'

    dev_files = traverse(root, 'dev')
    test_files = traverse(root, 'test')

    dev_positive_pairs, dev_negative_pairs = make_trials(dev_files, 100)
    test_positive_pairs, test_negative_pairs = make_trials(test_files, 100)

    write_trials(dev_positive_pairs, dev_negative_pairs, dev_trials)
    write_trials(test_positive_pairs, test_negative_pairs, test_trials)


if __name__ == '__main__':
    main()
