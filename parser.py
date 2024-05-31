import nltk

TERMINALS = """
Adj -> "armchair" | "country" | "day" | "home" | "little" | "red" | "the"
Adv -> "before" | "in"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the" | "this"
N -> "armchair" | "chuckled" | "companion" | "country" | "day" | "friend" | "holmes" | "home" | "paint" | "red" | "saturday" | "thursday" | "walk"
P -> "in" | "of" | "on" | "to" | "with"
V -> "arrived" | "chuckled" | "sat"
"""

NONTERMINALS = """
S -> NP VP | S Conj S
NP -> N | Det N | Det AdjP N | NP PP | AdjP NP
VP -> V | V NP | VP PP | VP AdvP | AdvP VP
AdjP -> Adj | AdjP Adj
AdvP -> Adv | AdvP Adv
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def preprocess(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words if any(char.isalpha() for char in word)]
    return words

def np_chunk(tree):
    chunks = []
    for subtree in tree.subtrees(lambda t: t.label() == 'NP'):
        if not any(child.label() == 'NP' for child in subtree):
            chunks.append(subtree)
    return chunks

def main():
    import sys
    from nltk import Tree

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            sentence = f.read()
    
    # Otherwise, get sentence as input
    else:
        sentence = input("Sentence: ")

    # Preprocess sentence
    words = preprocess(sentence)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(words))
    except ValueError as e:
        print(e)
        return

    # Print each tree
    for tree in trees:
        tree.pretty_print()

        # Print noun phrase chunks
        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))

if __name__ == "__main__":
    main()

