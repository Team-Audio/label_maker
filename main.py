import os
import re

import argh
import numpy as np
import tqdm as tqdm


def process_single_file(file: str, regex, sep):
    if isinstance(regex, str):
        regex = re.compile(regex)

    matches = regex.match(file)
    if matches is None:
        return np.array([])

    if sep is None:
        arr = np.array([int(x) for x in matches.groups()])
    else:
        groups = []
        for x in matches.groups():
            groups += list(x.split(sep))
        arr = np.array([int(x) for x in groups])

    return arr


def process_entire_directory(directory: str, regex, sep):
    regex = re.compile(regex)
    ret = {}
    for file in tqdm.tqdm(os.listdir(directory), desc='Walking files'):
        loc = os.path.join(directory, file)
        ret[loc] = process_single_file(file, regex, sep)

    return ret


@argh.named('file')
def file_wrapped(file: str, regex, sep=None, offs=0):
    print("To process:", file)
    print("Regex:     ", regex)
    if sep is not None:
        print("Seperator: ", sep)

    out = process_single_file(file, regex, sep)
    np.save(file + ".label", out - offs)


@argh.named('dir')
def directory_wrapped(directory: str, regex, sep=None, offs=0):
    print("To process:", directory)
    print("Regex:     ", regex)
    if sep is not None:
        print("Seperator: ", sep)

    res = process_entire_directory(directory, regex, sep)
    for key, value in tqdm.tqdm(res.items(), desc="Writing Labels"):
        if len(value) > 0:
            np.save(key + ".label", value - offs)


def main():
    parser = argh.ArghParser()
    parser.add_commands([file_wrapped, directory_wrapped])
    parser.dispatch()


if __name__ == '__main__':
    main()
