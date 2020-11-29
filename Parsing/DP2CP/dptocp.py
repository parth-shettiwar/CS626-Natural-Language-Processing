import spacy
import nltk

nlp = spacy.load('en')

list_sent = ['Three people laughed always',
'I play football',
'I like to drink',
'Car is running fast',
'India is best country',
'We work hard',
'I trust you',
'Winners never make excuse',
'You helped me',
'Strive for excellence']


list_sent2 = [
'I like to sing and dance',
'Fast dog caught the person',
'The quick brown fox jumps over the lazy dog',
'Big vehicles block the city road',
'I am testing the accuracy of parse tree',
'Blunders have occured when we are careless',
'I pretend to laugh always',
'It is a sunny day and I was sleeping',
'There was a jaguar behind the car',
'The fast athlete won the race'
]

labels = {"PART": "VP", "NP":"NP", "VP":"VP", "NOUN":"NP", "VERB":"VP","AUX":"VP","ADP":"PP", "DET":"NP", "ADV":"ADVP",  "PRON" : "NP", "PROPN" : "NP", "ADJ" : "NP"}

def separator(sent):
    for token in sent:
        if token.dep_ == "ROOT":
            return token.i

def get_noun(sent):
    for token in sent:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ == "nsubj" and child.i < token.i :
                    return child


def noun_phrase(sent,word,limit):
    children_dep = [x for x in word.children]
    if len(children_dep) == 0:
        return nltk.Tree(labels[word.pos_],[word.text]) if word.dep_ == 'nsubj' else nltk.Tree(word.pos_,[word.text]) 
    right_tree = []
    left_tree = []
    for c in children_dep:
        if c.dep_ == "punct":
            continue
        if c.i > limit:
            continue
        left_tree.append(noun_phrase(sent,c, limit)) if c.i < word.i else right_tree.append(noun_phrase(sent,c, limit))
    child =  [nltk.Tree("NP",left_tree + [nltk.Tree(word.pos_,[word.text])])] + right_tree if len(left_tree) else (left_tree + [nltk.Tree(word.pos_,[word.text])] +right_tree)
    return nltk.Tree(labels[child[0].label()],child)

def verb_phrase(sent,word):
    children_dep = [x for x in word.children]
    if len(children_dep) == 0:
        return nltk.Tree(word.pos_,[word.text])
    right_tree = []
    left_tree = []
    for c in children_dep:
        if c.dep_ == "punct":
            continue
        if c.i > word.i:
            right_tree.append(verb_phrase(sent,c))
        elif c.dep_ == "aux" or c.dep_ == "det" :
            left_tree.append(verb_phrase(sent,c))
    child = left_tree + [nltk.Tree("VP",[nltk.Tree(word.pos_,[word.text])]+right_tree)] if len(left_tree) else (left_tree + [nltk.Tree(word.pos_,[word.text])] +right_tree)
    return nltk.Tree(labels[child[0].label()],child)

for sent_ in list_sent2 : #["Work was not completed by him"]:
    sent = nlp(sent_)
    for token in sent:
        print(token.text, token.dep_, token.head.text, token.i,[x for x in token.children], token.pos_)
    tree = nltk.Tree("S",[noun_phrase(sent,get_noun(sent),separator(sent)),verb_phrase(sent,sent[separator(sent)])])
    tree.pretty_print()
