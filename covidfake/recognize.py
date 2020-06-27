import spacy, pickle, os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, FastText

model_path = f'{os.getcwd()}/cluster_models/model_3'
nlp = spacy.load('en_core_web_lg')
gm_model = pickle.load(open(f'{model_path}/gm.sklearn', 'rb'))
class_map = {
    3: 'TREATMENT/VACCINE',
    2: 'INSTITUTE',
    1: 'SCIENTIFIC',
    0: 'OTHER',
}

## If you'r going to use models 2 and 4 use FastText instead
embedings_model = Word2Vec.load(f'{model_path}/embedings_model.genism')

preproces_text = lambda doc: [ token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_space and not token.is_punct
    ]

def __get_entities(doc): 
    entities = pd.DataFrame(
        pd.Series(doc.ents),
        columns=['entitie']
    ).dropna()
    entities['label'] = entities.entitie.apply(lambda ent: ent.label_)
    entities['text'] = entities.entitie.apply(lambda ent: ent.text.lower())

    entities = entities[
        (entities.text.str.len() > 3)
        # & entities.label.isin(['ORG', 'LAW', 'PERSON', 'WORK_OF_ART', 'PRODUCT'])
        & ~entities.label.isin(['DATE', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])
    ].reset_index(drop=True)
    return entities

def __process_entiti(ent):
    tokens = preproces_text(ent)
    vectors = np.array([model_3[t] for t in tokens if t in model_3])
    return vectors.mean(axis=0)

def __ent_vector(ent):
    tokens = preproces_text(ent)
    vectors = np.array([embedings_model[t] for t in tokens if t in embedings_model])
    return vectors.mean(axis=0)

def recognize_entities(text):
    doc = nlp(text)
    ## Train with new corpus to ensure we have the vector embedings
    tokens = preproces_text(doc)
    embedings_model.train([tokens], total_examples=1, epochs=1)

    ## get entities and its vectors
    entities = __get_entities(doc)
    entities['vector'] = entities.entitie.apply(__ent_vector)
    entities = entities.dropna()
    
    vectors = entities['vector'].apply(pd.Series)
    entities['labels'] = gm_model.predict(vectors)
    entities['type'] = entities['labels'].apply(lambda n: class_map[n]) 

    return list(entities[['text', 'type']].T
        .to_dict()
        .values()
    )

    # return tokens
