import re
import json

"""
Raw sks_output here: https://medinfo.dk/sks/dump.php

Make two files
1) encoding json (mapping between ints and diagnose)
2) diagnose + diagnosetext
"""

with open('diagGPT/utils/sks_output.txt') as f:
    lines = f.readlines()

encoder = {}
diagnose_info = {}

for i, d in enumerate(lines[6:]):
    print(i, re.split("[\n\t]", d))
    d_split = re.split('[\n\t]', d)
    encoder[d_split[0]] = i
    diagnose_info[d_split[0]] = d_split[1]


with open('diagGPT/encoder.json', 'w') as fp:
    json.dump(encoder, fp, indent=-1)


with open('diagGPT/diagnose_info.json', 'w') as fp:
    json.dump(diagnose_info, fp, indent=-1)
