"""
Splits the large FineWeb text file into training and testing sets.
"""
from tqdm import tqdm

test_lines = 8222001
in_file = "data/fineweb.txt"
out_test = "data/fineweb_test.txt"
out_train = "data/fineweb_train.txt"


with open(in_file) as file, open(out_test, 'w') as out_test, open(out_train, 'w') as out_train:
    for idx, line in tqdm(enumerate(file)):
        if idx < test_lines:
            out_test.write(line)
        else:
            out_train.write(line)
