import argparse
from bs4 import BeautifulSoup
import re
import requests
import os
import sys

NOISE = True
URL = 'http://cs156.caltech.edu/scoreboard'

def run(fname):
    payload = {'teamid': 'ckbslftg', 'valset': 1}
    file_ = {'file': (fname, open(fname, 'rb'))}
    r = requests.post(URL, data=payload, files=file_)
    return scrape(r.text, fname)

def scrape(html, fname):
    soup = BeautifulSoup(html, 'html.parser')
    # 'Your current submission ... '
    text = soup.find_all('h3')[1].get_text()

    rmse = None
    regex = re.search('RMSE: (.+?) .*', text)
    if regex:
        rmse = float(regex.group(1))
        sys.stderr.write('{}: {}\n'.format(fname, regex.group(0)))
    return rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='file', type=str, nargs='+')
    args = parser.parse_args()

    for f in args.files:
        if NOISE:
            print('Adding noise to {}'.format(f))
            if not os.path.isfile('./bin/noise'):
                print('Need to build ./bin/noise! Exiting...')
                sys.exit()
            os.system('./bin/noise {}'.format(f))
            f += '_noisy.txt'

        print ('Uploading {}'.format(f))
        rmse = run(f)
        print ('{} rmse: {}'.format(f, rmse))
