import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from calendar import weekday, day_name

df = pd.read_csv("crime.csv")

# One of the TYPE's in the dataset is significantly more common than others so to even out distibution we drop all of those with an even index
typesWithLength = [(i,len(df[df.TYPE == i])) for i in df.TYPE.unique()]
mostCommonType = max(typesWithLength, key=lambda x: x[1])[0]
dropindexes = []
for index, row in df[df.TYPE == mostCommonType].iterrows():
    dropindexes.append(index)
dropindexes = list(filter(lambda x : x % 2 != 0, dropindexes))
df.drop(df.index[dropindexes], inplace=True)

# Clean up the dataset by dropping rows where they have NaN values

df.dropna(how='any',inplace=True)

# Initializing empty columns in the dataset so that we can fill them in the loop below

df["WEEKDAY"] = None
df["HUNDRED"] = None
df["BLOCK"] = None

"""
In this loop we fill the WEEKDAY column with the day of the week in the date 
that was in the dataset beforehand. We also split the HUNDRED_BLOCK into two different 
columns, one denoting the block number and another denoting the street name
"""
for index, row in df.iterrows():
    df.set_value(index, "WEEKDAY", day_name[weekday(row["YEAR"] ,row["MONTH"],row["DAY"])])
    # Here we split the HUNDRED_BLOCK string by spaces into a list
    hundredBlock = row["HUNDRED_BLOCK"].split()

    # We then extract the first element that includes an X since the data is on the form 45XX <STREETNAME>
    try:
        hundred = next((i for i in hundredBlock if "X" in i))
    except:
        hundred = "0"

    # Then we extract any digits from the extracted chunk but give it 0 if none is found
    hundred = 0 if not any(str.isdigit(d) for d in hundred) else int(''.join(filter(str.isdigit, hundred)))

    # Then we join the rest of the chunks together to construct the street name
    block = ' '.join([i for i in hundredBlock if "X" not in i])
    df.set_value(index, "HUNDRED", hundred)
    df.set_value(index, "BLOCK", block)

# We would like to combine the vehicle collisioin records into a single type since one of them only has ~300 values and the other ~40000

df.loc[df.TYPE.str.contains("Vehicle Collision"),"TYPE"] = "Vehicle Collision"
    
# These columns will not be used in this project
df.drop(["HUNDRED_BLOCK", "MINUTE","YEAR", "MONTH", "X","Y"],axis=1, inplace=True)

"""
In this loop we create a dictionary that calculates the average block number for a given street 
so in cases where we know the street name but not the number we can give that record the average 
street number
"""
averageStreetDictionary = {}
for index, row in df.iterrows():
    if '/' not in row["BLOCK"]:
        if row["BLOCK"] in averageStreetDictionary:
            val = row["HUNDRED"]
            prevAverage, prevCount = averageStreetDictionary[row["BLOCK"]]
            averageStreetDictionary[row["BLOCK"]] = ((((prevAverage*prevCount)+val)//(prevCount+1)), prevCount+1)
        else:
            
            averageStreetDictionary[row["BLOCK"]] = (row["HUNDRED"],1)
            
# We need the names of the columns for when we're appending rows to the dataset later
dfColumns = [col for col in df.columns]
"""
Here we iterate through all the rows that represent a crime that 
happened on a street corner and we split that into two records in 
the dataset each representing a crime on each street on the corner.
"""
dfToAppend = []
for index, row in df[df.BLOCK.str.contains("/")].iterrows():
    corners = [i.strip() for i in row["BLOCK"].split("/")]
    df.set_value(index,"BLOCK",corners[0])
    if corners[0] in averageStreetDictionary:
        df.set_value(index,"HUNDRED",averageStreetDictionary[corners[0]][0])
    dfToAppend.append([row[i] for i in dfColumns])
    dfToAppend[-1][dfColumns.index("BLOCK")] = corners[1]
    if corners[1] in averageStreetDictionary:
        dfToAppend[-1][dfColumns.index("HUNDRED")] = averageStreetDictionary[corners[1]][0]
    else:
        dfToAppend[-1][dfColumns.index("HUNDRED")] = 0
df = df.append(pd.DataFrame(dfToAppend, columns=dfColumns))


# Here we drop records where the values aren't on a format that is consitent with the rest of the dataset

df.drop(df.index[[354991]], inplace=True)

df.to_csv("crimesProcessed.csv", encoding='utf-8', index=False)
