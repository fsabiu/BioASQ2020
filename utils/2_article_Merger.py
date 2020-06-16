import time
import pandas as pd
import csv
import sys

def union(input, output):
    df_list = []
    l = 0
    # Time
    start = time.time()
    print(start)

    for dataset in inputs:
        df = pd.read_csv(dataset)
        l = l + len(df)
        try: 
            articles1 = df[['title', 'abstract', 'url', 'pubmed']]
        except:
            title = df[df.columns[7]]
            abstract = df[df.columns[17]]
            url = df[df.columns[13]]
            pubmed = df[df.columns[14]]

            frame = { 'title': title, 'abstract': abstract, 'url': url, 'pubmed':pubmed } 
            articles1 = pd.DataFrame(frame)

        # Check
        noTitle = articles1.title.isna().sum()
        if(noTitle>0):
            print("Warning: " + str(dataset) + " contatins null title(s)")

        noAbstract = articles1.abstract.isna().sum()
        if(noAbstract>0):
            print("Warning: " + str(dataset) + " contatins null abstract(s)")

        noPubMed = articles1.pubmed.isna().sum()
        if(noPubMed>0):
            print("Warning: " + str(dataset) + " contatins null url(s)")

        # Appending
        df_list.append(articles1)

    print(df_list)

    # Writing union
    finalDF = pd.concat(df_list)
    finalDF.to_csv(output)

    stop = time.time()

    # Check length
    if(len(finalDF)==l):
        print("Dataframes merged in " + str(stop) + "\nFinal dataframe in "+output)
    else:
        print("Error")


if __name__ == "__main__":
    print("leknflsk")
    n_files = len(sys.argv) - 1
    print("main")
    inputs = sys.argv[1:n_files]
    output = sys.argv[n_files]
    union(input, output)