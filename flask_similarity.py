from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
#import nltk
#from nltk.tokenize import sent_tokenize

class Similary():
    def __init__(self,arr_for_matching,arr_database,arr_preprocess,threshold,NER,Device):
        self.arr_for_matching = arr_for_matching
        self.arr_database = arr_database
        self.arr_preprocessing = arr_preprocess
        self.threshold = threshold
        self.NER = NER
        self.Device = Device
        #self.NLTK = NLTK
        if NER in ['ORG','PER','LOC']:
            self.bert_model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
            self.bert_tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')

    def preprocessing(self,arr, preprocessing_list):
        j = 0
        processed_arr = []
        for i in arr:
            clean_text = str(i)
            for k in preprocessing_list:
                clean_text = clean_text.replace(k, '')
            processed_arr.append(clean_text.strip().replace('  ', ' ').replace('  ', ' '))
            j = j + 1
        return processed_arr

    def find_max(self,arr):
        max_value = -1
        max_value_index = -1
        j = 0
        for i in arr:
            if i > max_value:
                max_value = i
                max_value_index = j
            j = j + 1
        return [max_value, max_value_index]

    def NER_analysis(self,article):
        if self.Device=='0':
            nlp = pipeline('ner', model=self.bert_model, tokenizer=self.bert_tokenizer, device=0)
        else:
            nlp = pipeline('ner', model=self.bert_model, tokenizer=self.bert_tokenizer, device=-1)
        ner_list = nlp(article)
        # print(ner_list)
        this_name = []
        all_names_list_tmp = []

        for ner_dict in ner_list:
            if ner_dict['entity'] == 'B-'+self.NER:
                if len(this_name) == 0 or '##' in ner_dict['word']:
                    this_name.append(ner_dict['word'])
                else:
                    all_names_list_tmp.append([this_name])
                    this_name = []
                    this_name.append(ner_dict['word'])
            elif ner_dict['entity'] == 'I-'+self.NER:
                this_name.append(ner_dict['word'])

        all_names_list_tmp.append([this_name])

        final_name_list = []
        for name_list in all_names_list_tmp:
            full_name = ' '.join(name_list[0]).replace(' ##', '').replace(' .', '.')
            final_name_list.append([full_name])
        if (final_name_list[0] == ['']):
            final_name_list = []
        del nlp
        return final_name_list

    def predict(self):
        external = self.preprocessing(self.arr_for_matching,self.arr_preprocessing)
        internal = self.preprocessing(self.arr_database,self.arr_preprocessing)
        model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        internal_list_embedding = model.encode(internal)
        external_list_embedding = model.encode(external)
        matches = []
        j = 0
        for i in external:
            entity_match = False
            if len(i) > 0:
                if self.NER not in ['ORG','PER','LOC']:
                    similarity = cosine_similarity([external_list_embedding[j]], internal_list_embedding)
                    [max_value, max_value_index] = self.find_max(similarity[0])
                    if max_value > self.threshold:
                        entity_match = True
                        matches.append({"best_match":self.arr_database[max_value_index], "match_ratio":str(max_value)})
                else:
                    external_entities = self.NER_analysis(i)
                    for entity in external_entities:
                        if entity_match == False:
                            similarity = cosine_similarity(model.encode(entity), internal_list_embedding)
                            [max_value, max_value_index] = self.find_max(similarity[0])
                            if max_value > self.threshold:
                                entity_match = True
                                matches.append({"best_match":self.arr_database[max_value_index], "match_ratio":str(max_value),
                                                "ner_entity":entity[0]})
                                break
            j = j + 1
        if entity_match == False:
            matches = [{"best_match":None, "match_ratio":str(0)}]
        del model
        return matches
