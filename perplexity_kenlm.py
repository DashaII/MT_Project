#!/usr/bin/env python

import os
import kenlm
import lzma
import re
import py7zr


def decompress(in_name, out_name, size=None):
    with lzma.open(in_name) as f_in, open(out_name, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if size is not None and i >= size:
                break
            f_out.write(line.decode('utf-8'))


def tokenize_file(from_file_path, to_file_path):
    with open(from_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Lowercase the text
    text = text.lower()

    # Replace non-breaking spaces with regular spaces
    text = text.replace('\u00A0', ' ')

    # Replace the � character with an empty string
    text = text.replace('�', '')

    # Separate punctuation with whitespace
    text = re.sub(r'([^\w\s])', r' \1 ', text)

    # Write the tokenized text back to the file
    with open(to_file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def decompress_split_files(split_files, output_dir):
    """
    Decompress each split file individually and save them in the output directory.
    The split files should be processed in sequence.
    """
    for split_file in split_files:
        print(f"Decompressing {split_file}...")
        with py7zr.SevenZipFile(split_file, mode='r') as archive:
            archive.extractall(path=output_dir)
        print(f"Finished decompressing {split_file}.")


if __name__ == '__main__':
    decompress('decompressed_en/en.00.deduped.xz.001', 'decompressed_en/en_2.txt')
    tokenize_file('decompressed_en\en_2.txt', 'decompressed_en\en_2_tokenized_manual.txt')

