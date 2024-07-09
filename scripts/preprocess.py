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
    df = pd.DataFrame()
    
    df['answers'] = pd.read_json(fname)['answers']
    df['answers'] = df['answers'].apply(helper)
    df['answers'] = df['answers'].apply(lambda x: sum(x, []))
    
    return df


if __name__ == "main":
    pass