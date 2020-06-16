import requests
import pandas as pd
from bs4 import BeautifulSoup
from csv import writer
import sys

invalid = []
# Columns
titles = []
abstracts = []
urls = []
pubmeds = []

def fetch(input, output):
    invalid = pd.DataFrame({}, columns = ['urls'])
    # Columns
    titles = []
    abstracts = []
    urls = []
    pubmeds = []

    # Reading urls
    unfetchedUrls = pd.read_csv(input)
    remaining = len(unfetchedUrls)

    for url in unfetchedUrls.URL:
        try:
            # HTTP request
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find(class_="rprt abstract").h1.get_text()
            abstract = soup.find(class_="abstr").div.get_text()
            pubmed = url
            # Appending
            titles.append(title)
            abstracts.append(abstract)
            pubmeds.append(pubmed)
            urls.append(pubmed)
            remaining = remaining - 1
            print(str(remaining) + " remaining")
        except:
            invalid = invalid.append({'urls': str(url)}, ignore_index= True)
            print(url + " invalid")
            remaining = remaining - 1
            print(str(remaining) + " remaining")
    
    # Creating dataframe
    frame = {'title': titles, 'abstract': abstracts, 'url': urls, 'pubmed':pubmeds} 
    lostArticles = pd.DataFrame(frame)

    # Writing dataframe
    lostArticles.to_csv(output)
    invalid.to_csv("invalid.csv")
    print(str(len(lostArticles)) + " articles saved")
    print(str(len(invalid)) + " invalid links")

if __name__ == "__main__":
    if(len(sys.argv)!=3):
        print ("Usage: " + sys.argv[0] + " <input_file.csv> <output_file.csv>\n") 
        sys.exit(1)

    input = sys.argv[1] # Input file
    output = sys.argv[2] # Input file
    fetch(input, output)