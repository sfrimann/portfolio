from SPARQLWrapper import SPARQLWrapper, JSON
import nltk
import pandas as pd
import numpy as np

def import_characters(verbose=False):
    """Import Harry Potter Characters from WikiData using SPARQL query"""

    ignored_characters = \
    ['Basilisk in the Chamber of Secrets', # The Chamber of Secrets bit artificially enhances presence
     'Fred Weasley II', # George Weasleys son. Does not appear in the book, but clashes every time Fred appears
     'Goyle senior', # Goyles' dad. Only appears in one scene, but clashes every time Goyle appears
     'Half-blood Prince', # Double entry. The Half-blood Prince is Severus Snape
     'Heir of Slytherin', # Double entry. This is Voldemort
     'Markus Belby', # Double entry of Marcus Belby
     'Regulus Arcturus Black', # Forefather of Regulus Black. Sirius' brother
     'Sirius Black I', # Forefather of Sirius Black. Difficult to separate from the one who appears in the books
     'Sirius Black II', # this one is Sirius Black II. Another descendant of the Sirius Black in the books
     'Tom Riddle'] # Voldemorts dad. Appears only a few times, and is very difficult to separate from Voldemort

    # SPARQL query to retrieve list of HP characters
    # query = """
    # SELECT DISTINCT ?item ?itemLabel ?itemAltLabel 
    #     WHERE 
    #     { 
    #          {
    #          ?item wdt:P31 ?sub1 .
    #          ?sub1 (wdt:P279|wdt:P131)* wd:Q95074 . 
    #          ?item wdt:P1080 ?sub2 . 
    #          ?sub2 (wdt:P279|wdt:P131)* wd:Q5410773 
    #          }
    #     OPTIONAL { ?item skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "en") }
    #     SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' . }
    #     }
    #     ORDER BY ?itemLabel
    # """

    query = """
    SELECT DISTINCT ?item ?itemLabel ?itemAltLabel 
        WHERE 
        { 
        {?item wdt:P31/wdt:P279* wd:Q15298221} UNION 
        {?item wdt:P31/wdt:P279* wd:Q154224} UNION
        {?item wdt:P31/wdt:P279* wd:Q15298259} UNION
        {?item wdt:P31/wdt:P279* wd:Q2087138} UNION
        {?item wdt:P31/wdt:P279* wd:Q15298195}.
        OPTIONAL { ?item skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "en") }
        SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' . }
        }
        ORDER BY ?itemLabel
    """

    # set up sparql connection to WikiData
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert() # run query and convert to json

    character_list = [] # initialize character list
    cid = 1
    for result in results["results"]["bindings"]: # loop over results
        # the character label (typically the name)
        label = result['itemLabel']['value']
        # alternative label. Not always present, but includes also known as.
        # Examples:
        #  - Alastor Moody a.k.a. Mad-Eye Moody
        #  - Arthur Weasley a.k.a. Mr. Weasley
        #  - Harry Potter a.k.a. Harry James Potter, The Boy Who Lived
        #  - Voldemort a.k.a. You-Know-Who, Tom Riddle, He-Who-Must-Not-Be-Named
        alt_label = result['itemAltLabel']['value'] if 'itemAltLabel' in result else ''

        if label[1:].isdigit():
            continue # Removes instances that are Q[number]. These are erroneous in WikiData

        # characters that I have manually decided to ignore
        if label in ignored_characters:
            continue

        # For each character I create a list of words associated with that character.
        # These words include the character name and a.k.a.
        # I remove stop words from the list of associated words (e.g. the in Beedle the Bard)
        # I also try to identify each character's first name (E.g. Harry, Hermione, Ron). I remove
        # prefixes like Mr., Mrs., and Sir.
        # Each character also gets the first letter from their first name associated with them (e.g.
        # H. for H. Potter)

        associated_words = label.split(' ') + [label[0]+'.']
        if alt_label:
            for aka in alt_label.split(', '):
                for word in aka.split(' '):
                    # if present remove quotes in word (sometimes used for nicknames)
                    associated_words += [word.replace('"', '').replace("'", "")]
        # make all associated words lowercase
        associated_words = [word.lower() for word in associated_words]
        # remove stopwords from associated words
        associated_words = [word for word in associated_words
                            if word not in nltk.corpus.stopwords.words('english')+['sir']]

        # identify character's first name
        first_name = associated_words[0] \
                     if associated_words[0] not in ['mr.', 'mrs.', 'sir'] \
                     else None

        # make associated words into set
        associated_words = set(associated_words)

        # make dict and append to characters list
        cdict = dict(cid=cid,
                     label=label,
                     associated_words=associated_words,
                     first_name=first_name)
        if verbose:
            print('id: {cid}. Character: {label}. First name: {first_name}. Associated Words: {associated_words}'.format(**cdict))
        character_list.append(cdict)
        cid += 1

    return character_list

def character_appearance_in_text(tokens, character_list, verbose=False):
    """Searches for words associated with each character amongst the token.
    If there is match the token and the character becomes associated.
    The algorithm is greedy in the sense that one tokens can be associated with
    several characters.
    The result is a matrix that maps tokens and characters. I make two matrices:
    One for all the words associated with the character, and one only for the
    first name.

    There are two circumstances where characters from import_characters are not
    included in the matrices:
      1. If they are not matched with any tokens
      2. If they are members of the Black, Malfoy, or Weasley family and their
         first name is not matched with any tokens.
    """

    character_appearance_list = []
    first_name_list = []
    for i, character in enumerate(character_list):
        # find tokens that can be associated with a character
        character_appearance = pd.Series(
            tokens.lower.apply(lambda x: x in character['associated_words']).astype(bool),
            index=tokens.index
            )
        # character names always begins with a capital letter. Checking for this avoids instances
        # where a character name is mistaken for regular word (e.g. Crouch in Barty Crouch)
        character_appearance = character_appearance & tokens.capital
        character_appearance.name = character['label']

        # do not include if character is not associated with any tokens
        casum = character_appearance.sum()
        if casum == 0:
            continue

        # find tokens that can be associated with a character's first name
        first_name_appearance = pd.Series(
            tokens.lower.apply(lambda x: x == character['first_name']).astype(bool),
            index=tokens.index
            )
        # character names always begins with a capital letter. Checking for this avoids instances
        # where a character name is mistaken for regular word (e.g. Crouch in Barty Crouch)
        first_name_appearance = first_name_appearance & tokens.capital
        first_name_appearance.name = character['label']

        # do not include if first_name is not associated with any token and character is a member
        # of the Black, Malfoy, or Weasley family
        fnsum = first_name_appearance.sum()
        if fnsum == 0:
            if character['associated_words'].intersection(['black', 'malfoy', 'weasley']):
                continue

        # add character_appearance and first_name_appearance to lists
        character_appearance_list.append(character_appearance)
        first_name_list.append(first_name_appearance)

        if verbose:
            print("{}. Character: {}. First name count: {}. Token count: {}".format(i,
                character['label'], fnsum, casum))

    character_appearances_df = pd.concat(character_appearance_list, axis=1)
    first_names_df = pd.concat(first_name_list, axis=1)

    return first_names_df, character_appearances_df

def mixed_characters(matrix):
    """Given character matrix, return overview over characters that are most often confused with
    each other
    """
    index, = np.where(matrix.sum(axis=1) > 1)
    mixed = []

    character_labels = matrix.columns.values

    for i in index:
        mixed.append(', '.join(character_labels[matrix.iloc[i].values]))

    return pd.Series(mixed, index=index)

def kernel(size, kind='triangular'):
    """Kernel function to help the character scoring"""
    if size % 2 == 0:
        raise ValueError("Kernel length must be odd")

    u = np.linspace(-1, 1, size)

    if kind == 'triangular':
        k = 1 - np.abs(u)
    elif kind == 'parabolic':
        k = 3/4*(1 - u**2)

    return k

def score_characters(text_matrix, first_name_matrix, character_matrix, dx=6, kind='triangular',
                     verbose=False):
    """Given character matrices, score each token to help determine which character it refers to.
    The scoring is done by lookint at a small section of tokens surrounding the central token, and
    labelling the characters the surrounding tokens refer to. A triangular kernel is used so that
    tokens close to the main token are weighted higher.
    If the main token is mr., mrs. or professor the following token is assigned a higher weight.
    If a token is in the first_name_matrix it is added the character_matrix with a weight of 0.2
    """

    k = kernel(2*dx+1, kind=kind) # kernel for score weights
    indices, = np.where(character_matrix.sum(axis=1) > 1) # tokens with more than one candidate character

    tokens = text_matrix.lower.astype(str)

    score_list = []
    for i, index in enumerate(indices): # loop over indices
        if verbose:
            print(i, index, indices.size)

        # incides around central token
        df_index = np.arange((index-dx), (index+dx+1), dtype=np.int)

        # weights array
        weight = np.ones(2*dx+1, dtype=np.float)
        for j in range(1, 2*dx+1):
            if tokens[df_index[j-1]][-1] == '.':
                weight[j] = 1.5
            if tokens[df_index[j-1]] == 'professor':
                weight[j] = 1.5

        factor = weight*k

        # apply weights
        score = (factor[:, np.newaxis]*character_matrix.iloc[df_index]).sum()
        score += 0.2*first_name_matrix.iloc[df_index].sum()

        score_list.append(score)

    score_df = pd.concat(score_list, axis=1).T
    score_df.index = indices

    return score_df

def popularity_contest(character_matrix, verbose=False):
    """Decide candidate characters by who appears the most times"""

    counts = character_matrix.sum(axis=0)
    # Some counts are enhanced artificially to make sure the final order
    # is right
    counts['Draco Malfoy'] += 100 
    counts['Nymphadora Tonks'] += 100
    counts['James Potter'] += 100

    character_labels = counts.index.values

    indices, = np.where(character_matrix.sum(axis=1) > 1)

    popularity_list = []
    for i, index in enumerate(indices):
        if verbose:
            print(i, index, indices.size)

        row = character_matrix.loc[index]
        clabel = character_labels[row.values]
        mx = counts[clabel].max()

        popularity_list.append((counts == mx) & row)

    popularity_df = pd.concat(popularity_list, axis=1).T
    popularity_df.index = indices

    return popularity_df

def character_group(character_matrix):

    character_labels = character_matrix.columns
    character_groups = []
    for character in character_labels:
        index, = np.where(character_matrix[character])
        diff = np.diff(index)

        if (index.size < 2):
            character_groups.append(character_matrix[character])
            continue
        if diff.min() > 1:
            character_groups.append(character_matrix[character])
            continue

        index = np.insert(index[1:][diff > 1], 0, index[0])

        character_appearance = pd.Series(np.zeros(character_matrix.shape[0], dtype=np.bool))
        character_appearance[index] = True
        character_appearance.name = character
        character_groups.append(character_appearance)

    return pd.concat(character_groups, axis=1)


characters = import_characters(verbose=True)

hp_dataframe = pd.read_pickle('hp.gz')

first_names, character_appearances = character_appearance_in_text(hp_dataframe, characters, verbose=False)

scores = score_characters(hp_dataframe, first_names, character_appearances, verbose=False)

scores_bool = scores.apply(lambda x: x == np.max(x), axis=1)

character_appearances.update(scores_bool)
character_appearances = character_appearances.astype(bool)

print(character_appearances.sum(axis=1).value_counts())

print(character_appearances.sum().sort_values(ascending=False).to_string())

popularity_bool = popularity_contest(character_appearances, verbose=False)

character_appearances.update(popularity_bool)
character_appearances = character_appearances.astype(bool)

print(character_appearances.sum(axis=1).value_counts())

print(character_appearances.sum().sort_values(ascending=False).to_string())

# for i in s.index:
#     print(s.loc[i])
#     print(hp_dataframe['token'].iloc[(i-6):(i+7)])