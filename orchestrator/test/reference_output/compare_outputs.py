import json
import ase.io
import numpy as np
from numpy import allclose, abs as nabs


def compare_outputs(ref_file: str, test_file: str, skip_lines: int = 0):
    """
    helper function for pytests to compare the content of two files

    iterates over files line by line splitting lines into space separated words
    to directly compare strings and compare floats within a tolerance of 1e-3%.
    Since words can include 'string=float', also break words into subwords
    split on '='. If an assert statement will fail, print the condition to
    stdout for reference.

    :param ref_file: path to the reference file to check against
    :type ref_file: str
    :param test_file: path the the test file to check
    :type test_file: str
    :param skip_lines: a number of lines to skip at the beginning of each file
    :type skip_lines: int
    """

    with open(ref_file, 'r') as ref_in, open(test_file, 'r') as test_in:
        for i in range(skip_lines):
            _ = ref_in.readline()
            _ = test_in.readline()
        ref_lines = ref_in.readlines()
        test_lines = test_in.readlines()
        assert len(ref_lines) == len(test_lines), \
            "Number of lines in the test and ref files don't match"

    for ref_line, test_line in zip(ref_lines, test_lines):
        # if a line has metadata we'll skip it as this can be different
        # for different users (mostly an issue with oracle output)
        has_metadata = False
        ref_words = ref_line.strip().split()
        test_words = test_line.strip().split()
        assert len(ref_words) == len(test_words), \
            "Number of words in the test and ref lines don't match" \
            f"ref line:\n{ref_line}test line:\n{test_line}"
        for ref_word, test_word in zip(ref_words, test_words):
            if has_metadata:
                break
            # some floats are joined by = with no spaces
            ref_subwords = ref_word.split('=')
            test_subwords = test_word.split('=')
            assert len(ref_words) == len(test_words), \
                "Number of subwords in the test and ref words don't match" \
                f"ref word:\n{ref_word}test word:\n{test_word}"
            for ref_subword, test_subword in zip(ref_subwords, test_subwords):
                # exyz files can include quotes as well, strip these
                ref_clean = ref_subword.strip().strip('\'"').strip()
                test_clean = test_subword.strip().strip('\'"').strip()
                if (test_clean == 'metadata' or test_clean == '_metadata' or
                    test_clean == 'quests_delta_entropy_reference_set'):
                    has_metadata = True
                    break
                try:
                    ref_num = float(ref_clean)
                    test_num = float(test_clean)
                    # if no ValueError, then values are numbers and
                    # should be compared up to tolerance
                    ref_test_diff = nabs(test_num - ref_num)
                    if ref_test_diff > 1e-10 + 5e-3 * nabs(ref_num):
                        print(f'Reference val: {ref_num}, Test val: '
                              f'{test_num}, diff: {ref_test_diff},'
                              f'limit:{1e-10 + 5e-3*nabs(ref_num)}')
                    assert allclose(
                        test_num,
                        ref_num,
                        rtol=5e-3,
                        atol=1e-10,
                        equal_nan=True,
                    )
                except ValueError:
                    # word is not a number, do string comparison
                    if ref_clean != test_clean:
                        print((f'Reference string: {ref_clean}, Test '
                               f'string: {test_clean}'))
                    assert ref_clean == test_clean


def compare_atoms_info(ref_file: str, test_file: str, compare_key: str):
    """
    helper function for pytests, especially made when the values to compare
    are stored in `atoms.info`.

    The issue with `compare_outputs()` in this specific case is that it ignores
    everything after `METADATA_KEY` on the same line. However, ASE actually
    does save some result values after `METADATA_KEY`, which are being
    overlooked.

    :param ref_file: path to the reference file to check against
    :type ref_file: str
    :param test_file: path the the test file to check
    :type test_file: str
    :param compare_key: key in `atoms.info` that contains values to compare
    :type compare_key: str
    """
    # Read the files as atoms
    ref_atoms_list = ase.io.read(ref_file, index=':')
    test_atoms_list = ase.io.read(test_file, index=':')
    # Iterate over atoms
    for ref_atoms, test_atoms in zip(ref_atoms_list, test_atoms_list):
        # Get the values to compare
        ref_value = ref_atoms.info[compare_key]
        test_value = test_atoms.info[compare_key]
        ref_test_diff = nabs(test_value - ref_value)
        if np.any(ref_test_diff > 1e-10 + 5e-3 * nabs(ref_value)):
            print(f'Reference val: {ref_value}, Test val: '
                  f'{test_value}, diff: {ref_test_diff},'
                  f'limit:{1e-10 + 5e-3*nabs(ref_value)}')
        assert allclose(ref_value,
                        test_value,
                        atol=1e-10,
                        rtol=5e-3,
                        equal_nan=True)


def compare_json(ref_file: str, test_file: str, compare_key: str):
    """
    helper function for pytests, especially made when the values to compare
    are stored in a JSON file.

    As mentioned in `compare_outputs()`, we want to avoid comparing metadata
    due to its inconsistent format. However, when the result is saved as a
    JSON file containing metadata, the metadata will be written across multiple
    lines, causing `compare_outputs()` to overlook or misinterpret the metadata
    when performing comparisons.

    :param ref_file: path to the reference file to check against
    :type ref_file: str
    :param test_file: path the the test file to check
    :type test_file: str
    :param compare_key: key in the dictionary read from JSON file that contains
        values to compare
    :type compare_key: str
    """
    # Read the files
    with open(ref_file, 'r') as f:
        ref_dict = json.load(f)
    with open(test_file, 'r') as f:
        test_dict = json.load(f)
    # Get the values to compare
    ref_value = np.array(ref_dict[compare_key])
    test_value = np.array(test_dict[compare_key])
    ref_test_diff = nabs(test_value - ref_value)
    if np.any(ref_test_diff > 1e-10 + 5e-3 * nabs(ref_value)):
        print(f'Reference val: {ref_value}, Test val: '
              f'{test_value}, diff: {ref_test_diff},'
              f'limit:{1e-10 + 5e-3*nabs(ref_value)}')
    assert allclose(ref_value,
                    test_value,
                    atol=1e-10,
                    rtol=5e-3,
                    equal_nan=True)
