import json
import csv

def fetchURLs():
    # Reading dataset
    with open("data/training8b.json") as f:
        d = json.load(f)

    # Urls dictionary
    urls = dict()

    # Adding unique
    for question in d["questions"]:
        for url in question["documents"]:
            try:
                urls[url] = urls[url] + 1
            except:
                urls[url] = 1

    # Checking length
    len(urls)

    # Writing URLs
    csv_file = "data/urls.csv"
    with open(csv_file, 'w') as f:
        f.write("%s,%s\n"%("URL","Occurrencies"))
        for key in urls.keys():
            f.write("%s,%s\n"%(key,urls[key]))

if __name__ == "__main__":
    fetchURLs()
