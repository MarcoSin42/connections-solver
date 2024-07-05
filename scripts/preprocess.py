import os
import pandas as pd

def helper(row):
    out = []
    for level in row:
        out.append(
            [member.lower() for member in level['members']]
        )
    return out

def parse_json(fname:str="../NYT-Connections-Answers/connections.json"):
    df = pd.read_json(fname)['answers']
    
    return df.apply(lambda x: helper(x))



if __name__ == "main":
    load_json()
    