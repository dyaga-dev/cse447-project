#!/usr/bin/env python
import os
import re

TEST_DIR = "testing_data"
OUT_INPUT = os.path.join(TEST_DIR, "all_test_input.txt")
OUT_ANSWER = os.path.join(TEST_DIR, "all_test_answer.txt")

# Collect language codes from input files
input_files = []
pattern = re.compile(r"test_input_(.+)\.txt")

for fname in os.listdir(TEST_DIR):
    match = pattern.match(fname)
    if match:
        lang = match.group(1)
        input_files.append((lang, fname))

# Sort by language code for deterministic order
input_files = sorted(input_files, key=lambda x: x[0])

with open(OUT_INPUT, "w", encoding="utf-8") as fout_in, \
     open(OUT_ANSWER, "w", encoding="utf-8") as fout_ans:

    for lang, input_fname in input_files:
        answer_fname = f"test_answer_{lang}.txt"

        input_path = os.path.join(TEST_DIR, input_fname)
        answer_path = os.path.join(TEST_DIR, answer_fname)

        if not os.path.exists(answer_path):
            raise ValueError(f"Missing answer file for language: {lang}")

        # Write input lines
        with open(input_path, encoding="utf-8") as fin:
            for line in fin:
                fout_in.write(line)

        # Write answer lines
        with open(answer_path, encoding="utf-8") as fan:
            for line in fan:
                fout_ans.write(line)

print("Successfully combined test files.")