import json
import re
import random
from flair.data import Sentence
import progressbar
import pickle
from sklearn.model_selection import train_test_split

############################################################LIST
def load_data_list(csv_file):
    # file di input per il json
    # tipo di domande da recuperare

    with open(csv_file, "r", errors = 'ignore') as read_file: # After analysing errors we decided to ignore them
        training = json.load(read_file)
    output = [[question["body"], question["exact_answer"], [snippet["text"] for snippet in question["snippets"]]]
              for question in training["questions"] if question["type"] == "list"]
    return output

def load_data(csv_file, questionType, singleSnippets = False):
    """
    Parameters:
    - csv_file: JSON input file
    - questionType: type of the questions to be retrieved
    - singleSnippets: boolean value specifying whether <question, snippet> pairs must be returned

    Return:
    - list of <body, exact answer, ideal answer, snippet list> for each question of the questionType type.
    """
    if(questionType not in ["factoid", "list", "yesno"]):
      raise Exception("Unknown question type: " + questionType)
    
    # Opening training set
    with open(csv_file, "r", errors = 'ignore') as read_file:
        training = json.load(read_file, encoding='utf-8')

    output = []
    append = output.append
    
    if(singleSnippets == False):
      try:
        output = [[question["body"], question["exact_answer"], question["ideal_answer"], [snippet["text"] for snippet in question["snippets"]]]
                  for question in training["questions"] if question["type"] == questionType]
      except:
          pass
    else:
      for question in training["questions"]:
        try:
          for snippet in question["snippets"]:
            if question["type"] == questionType:
              try:
                append({"body":question["body"], "exact_answer":question["exact_answer"], "ideal_answer": question["ideal_answer"], "snippet": snippet})
              except:
                print("Missing fields")
        except:
          pass
    return output

def clean_synonyms(list_question):
  #Le funzione prende la lista nativa di domande e ricerca i vari sinonimi.
  #Cancella tutti i sinonimi e prende solo il primo valore conservano i sinonimi in un dizionario
  #La funzione restituisce i dati puliti e un dizionario dei sinonimi
  dict_syn={}

  for question in list_question:
    for answer in question[1]:
      if(len(answer)>1):
        for elem in answer:
          new_list=[other_elem for other_elem in answer if other_elem !=elem]
          new_syn=set(new_list)
          if(elem not in dict_syn):
            dict_syn[elem]=new_syn
          else:
            dict_syn[elem].union(new_syn)

  return list_question,dict_syn

##################################################################YESNO
def load_data_yesno(csv_file):
    # file di input per il json
    # tipo di domande da recuperare

    with open(csv_file, "r") as read_file:
        training = json.load(read_file)
    output = [[question["body"], question["exact_answer"], [snippet["text"] for snippet in question["snippets"]]]
              for question in training["questions"] if question["type"] == "yesno"]
    return output




def generate_embeddings_yesno(model, data, file_name):

    data_embeddings = []

    bar = progressbar.ProgressBar(max_value=len(data)-1).start()
    
    for i, question in enumerate(data):
        sample = []
        #Question
        sample.append(model.embed(Sentence(question[0])))
        #Answer
        sample.append(question[1])
        #Snippets
        snippet_list = []
        for snippet in question[2]:
            if(len(snippet.split())<400):
                snippet_list.append(model.embed(Sentence(snippet)))
            else:
                 snippet_list.append("__Failed__")
        sample.append(snippet_list)
        data_embeddings.append(sample)
        bar.update(i)

    dbfile = open(file_name, 'ab')
    pickle.dump(data_embeddings, dbfile)
    dbfile.close()


def generate_embeddings_yesno_pooling(model, data, filename=None):
    
    data_embeddings = []

    bar = progressbar.ProgressBar(maxval=len(data)-1).start()
    for i, sample in enumerate(data):
        sample_emb = []
        #Question
        question=Sentence(sample[0])
        model.embed(question)
        sample_emb.append(question.get_embedding())

        #Answer
        encoder = {'yes':1, 'no':0}
        sample_emb.append(encoder[sample[1]])

        #Snippets
        snippet_list = ""
        for snippet in sample[2]:
            snippet_list+=snippet

        # Creo Sentences
        snippets_sent=Sentence(snippet_list)
        model.embed(snippets_sent)

        # ottengo tensore (?)
        sample_emb.append(snippets_sent.get_embedding())
        data_embeddings.append(sample_emb)
        bar.update(i)
    bar.finish()

    if filename is not None:
      dbfile = open(filename, 'ab')
      pickle.dump(data_embeddings, dbfile)
      dbfile.close()
      
    return data_embeddings

def load_embeddings(file_name):
    dbfile = open(file_name, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    return data

def toYesNo(e):
  """
  Returns 'yes' or 'no' depending on the input e.
  e parameter can be either a numpy array or a single element
  """
  import numpy as np

  # None
  if e is None:
    return None

  # String input
  if(isinstance(e, np.str_) or (isinstance(e, np.ndarray) and len(e) > 0 and isinstance(e[0], np.str_))):
    raise Exception('Type Error: toYesNo only takes numbers, not strings!')

  # Array or value
  if isinstance(e, np.ndarray):
    return np.array(['yes' if value >= 0.5 else 'no' for value in e])
  else:
    if(isinstance(e, np.int64) or isinstance(e, int) or isinstance(e, float)):
      return 'yes' if e >= 0.5 else 'no'

def toNumbers(e):
  """
  Returns 1 or 0 depending on the input string.
  e parameter can be either a numpy array or a single element
  """
  import numpy as np

  # None
  if e is None:
    return None

  # Array or value
  if (isinstance(e, np.str_) or isinstance(e, np.ndarray)):
    return np.array([1 if (value == 'yes' or value == 'Yes' or value == 'y' or value == 'Y') 
                           else 0 for value in e])
  else:
    if(isinstance(e, str)):
      return 1 if (e == 'yes' or e == 'Yes' or e == 'y' or e == 'Y') else 0

  return None

def data_split(X, y, y_size):
  """
  Returns X_train,X_test,y_train,y_test depending on the split ratio y_size
  """
  return train_test_split(X,y,test_size=y_size)

def find_sub_list(sl,l):
  """
  Parameters:
  - sl is a list of strings (words)
  - l is a list of strings following the pattern a::['SEP']::b with a and b ebentually being empty.

  Returns:
  - A list of all the initial and final indices (tuples) of l at which sl starts and ends, if any.
  - Else, empty list
  """
  results=[]
  sll=len(sl)
  for ind in (i for i,e in enumerate(l) if e==sl[0]):
      if (l[ind:ind+sll]==sl and ind>l.index('[SEP]')):
          results.append((ind,ind+sll-1))
  return results

def getsubIndices(subs, s):
  """
  Parameters:
  - s: a string
  - subs: the pattern to be identified in s

  Returns:
  - A list of the initial and final position(s) at which subs occurs in s, if any.
  - An empty list, in any other cases.
  """
  # Tokenization
  pattern = re.sub("[^\w]", " ",  subs).split()
  seq = re.sub("[^\w]", " ",  s).split()

  return find_sub_list(pattern, seq)


def yesNoAugmentation(target, n_questions, singleSnippets):
  """
  Parameters:
  - target: the target of the required questions
  - n_questions: the number of required questions
  - singleSnippets: boolean value specifying whether <question, snippet> pairs must be returned

  NOTICE that the parameter n_questions specifies the number of questions, independently from the number of snippets
  (important if single snippets are required) 

  Returns:
  - A list of "n_questions" <body, exact answer, ideal answer, snippet list>, randomly selected among those having the specified target.
  """
  fifth = load_data("../../data/training5b.json", "yesno", singleSnippets)
  sixth = load_data("../../data/training6b.json", "yesno", singleSnippets)
  seventh = load_data("../../data/training7b.json", "yesno", singleSnippets)

  # Filter records with target = target (yes/no)
  
  questions = []
  if(singleSnippets):
    questions = [q for q in fifth + sixth + seventh if q["exact_answer"] == target]
  else:
    questions = [q for q in fifth + sixth + seventh if q[1] == target]

  # Total number of questions
  total = len(questions)

  # Checking available questions number
  if (n_questions > total):
    raise Exception("Not enough questions in the three previous datasets: required " + str(n_questions) + " of " + str(total))
  
  # Sampling n_questions different questions (or not)
  if (n_questions == -1):
    questions_indices = range(len(questions))
  else:
    questions_indices = random.sample(range(0, total-1), n_questions)

  result = []

  for idx in questions_indices:
    result.append(questions[idx])

  return result
