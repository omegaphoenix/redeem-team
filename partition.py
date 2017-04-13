"""
Partition data into
    1.dta - general data (96%)
    2.dta - general data (2%)
    3.dta - general data (2%)
    4.dta - probe set
    5.dta - same as qual, used to test this script.
"""

def partition(dir_name, part):
    """Create a data partition.

    dir_name - name of subdirectory within data/
    part - integer number of part
    """
    data_file = 'data/{}/all.dta'.format(dir_name)
    part_file = 'data/{}/all.idx'.format(dir_name)
    output_file = 'data/{}/{}.dta'.format(dir_name, part)

    with open(data_file) as data, open(part_file) as parts, \
         open(output_file, 'w') as output:
        for line in data:
            cur_part = parts.readline()
            if int(cur_part) == part:
                output.write(line)

if __name__ == '__main__':
    for dir_name in ['mu', 'um']:
        for part in range(1,6):
            partition(dir_name, part)
