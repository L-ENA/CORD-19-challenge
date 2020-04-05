# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:09:13 2019

@author: xf18155
"""

import pandas as pd
from collections import Counter
import re
import random
import nltk

nltk.download('punkt')
import json
import numpy as np
from nltk.tokenize import sent_tokenize
from glob import glob
from tqdm import tqdm
import collections
from collections import defaultdict
import os


class NpEncoder(
    json.JSONEncoder):  ##as json can not serialise obj of type int64. this class converts some common problematic datatypes
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def makeInput(paths, types="P", data_link = ""):
    #
    #This method basically reads in our data that we want to mine. Originally it was developed for pubmed abstracts only (hence PMID column name,
    # but now we will just use the PMID column with cord-id content)
    #
    #

    versionString = "v2.0"  # squad formalities
    data = {'version': versionString, 'data': []}  # to have possibility of pure new squad training data

    def addAbsData(sents, topicString):  # add mining questions for one abstract
        # quests=['Which intervention did the patients receive?','Which intervention did the participants receive?', 'What was the intervention?', 'What did the patients receive?', 'What did the participants receive?', 'What was the intervention given to the participants?']
        if types == "P":
            quests = ['Who was treated?', 'What were the criteria for enrollment?', 'What were the inclusion criteria?',
                      'Who was enrolled in the study?', 'What were participants diagnosed with?',
                      'Which participants took part in the study?']

        elif types=="I":
            quests = ['Which intervention did the patients receive?',
                      'Which intervention did the participants receive?', 'What was the intervention?',
                      'What did the patients receive?', 'What did the participants receive?',
                      'What was the intervention given to the participants?']

        else:
            quests = ['Which condition did the participants have?',
                      'Patients with which medical conditions were included?', 'What was the medical condition?',
                      'What was the condition?']

        # quests = ['Which condition did the participants have?', 'Patients with which medical conditions were included?', 'What was the medical condition?', 'What was the condition?']
        # quests = ['What age were the participants?', 'How old were the participants?', 'What age were the patients?', 'How old were the patients?', 'What was the age in the population?']
        quest = random.sample(quests, k=1)[0]  # as this returns list, we take first item of list###random question

        domainDict = {'title': topicString, 'paragraphs': []}  ##append many, change this to constructor
        for context in sents:

            try:
                domainDict['paragraphs'].append({'qas': [], 'context': re.sub('([^\s])(\.)', r'\1 \2', context)})
            except:
                domainDict['paragraphs'].append({'qas': [], 'context': " "})
        counter = 0
        for index, paragraph in enumerate(domainDict['paragraphs']):
            qaDict = {'question': quest, 'id': '{}_{}'.format(topicString, counter), 'answers': [],
                      'is_impossible': True}  # as we dont have and labels here
            if len(paragraph['context'].split(' ')) < 3:
                paragraph['context'] = 'NA abstract is missing'
            qaDict['plausible_answers'] = [
                {
                    "text": ' '.join(paragraph['context'].split(' ')[:2]),  # first 2 words
                    "answer_start": 0
                }
            ]

            paragraph['qas'].append(qaDict)

            counter += 1  # advance counter for individual ids on sentence contexts

        return domainDict

    sent_df = pd.DataFrame(
        columns=['ID', 'Sent', 'Participant_Pred', 'Condition_Pred', 'Age_Pred', 'Outcome_Pred', 'Intervention_Pred'])# some df that could store all PICO mined data as central place
    row_list = []

    for path in tqdm(paths, total=len(paths)):
        #if you have a bigger GPU than me then you can read in more files per folder :)
        # #My initial testing dataset on immunoglobulin was 120.00+ abstracts so with 10.000 abstracts per file I had 6 batches with 2 files.
        # #This was hard because Colab (10-12GB GPU?) kept getting out of error or runtime disconnects when predicting.

        df = pd.read_csv(path)

        for index in df.index:
            # print(index)
            try:  # tokenize abstract into sents
                sents = sent_tokenize(df['Abstract'][index])
            except:  # there is no abstract
                sents = []

            sents.insert(0, df['Title'][index])  # add title sa first sentence
            counter = 0
            ####squad conversion
            topicString = df['PMID'][index]
            if len(str(
                    topicString)) < 5:  #luckily cord hs all ids. This code is only important if one mines data from pubmed abstracts with missing pmids# some cases where we have no pmid: since we need unique ids, we have to make up a new id. otherwise, predictions will be lost: around 65 sentences in first abstracts file
                topicString = 'no_id_available_index_' + str(index) + "_2"
                print(topicString)

            domainDict = addAbsData(sents, topicString)
            data['data'].append(domainDict)

            for sent in sents:
                idNew = '{}_{}'.format(topicString, counter)
                row_list.append({'ID': idNew, 'Sent': sent})
                counter += 1

    # length 1st batch: 1746010
    sent_df = pd.DataFrame(row_list)
    #sent_df.to_pickle(
        #'backup_pre_prediction_'+ types+ '.pkl')
    sent_df.to_csv(
        os.path.join('predictions', 'sent_map.csv'))# will add predictions later

    #uncomment next few lines if you want to add some orig. squad data to the evaluation!
    #with open(
    #        "C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Sarah P mining\\immunoglobulin\\squadInput\\dev-v2.0.json",
    #        encoding='utf-8') as feedsjson:
    #    feeds = json.load(feedsjson)
    #    print(len(feeds['data']))

    #    data['data'] = data['data']  # +feeds['data']

    with open(
            'dev-v2.0_cord_pred_' + types +'.json',
            mode='w') as f:
        f.write(json.dumps(data, indent=2, cls=NpEncoder))

    print(len(data['data']))


def process_input(preds, ids, mode):
    #Once prediction on colab is finished One can do some minor post-procesing on the records in order to cluster them better
    #
    #

    preds = [p.lower() for p in preds]  # lower for easier deduplication
    preds = [re.sub(r'[“”"‘’\'\[\]{}«»„‚‹›»«]', "", p).strip() for p in preds]#no need for those chars, especially for all incomplete parentheses
    preds = [re.sub(r'^(\d+\s)?patients? ((treated )?with|receiving|undergoing)', '', p).strip() for p in preds]#common sentence beginnings dont add mich either. Note ^ as anchor, omitting this leads to strange results where middle of senences are missing.
    preds = [re.sub(r'^ (\d+[., ]?) + (\s?(cases?( of)? | patients?(with)))', '', p).strip() for p in preds]
    preds = [re.sub(r'^ (\d+[., ]?) + (\s?(cases?( of)? | patients?(with)))', '', p).strip() for p in preds]
    preds = [re.sub(r'^\d+\s(patient|participant|subject)s?[,.:;]?', '', p).strip() for p in preds]#common sentence beginnings dont add mich either
    preds = [re.sub(r'^(covid[ -]?19|covid|coronavirus infectious disease( -19)?|(2019[ -]?)?novel[ -]?corona[ -]?virus|coronavirus|2019[ -]?ncov|sars[ -]?cov[ -]?2|corona[ -]?virus[ -]?disease([ -]?2019)?)', '', p).strip() for p in preds]#these commin beginnings of mined results do not add any value - I already know that my population has corona
    preds = [re.sub(r'^(patients|in |pandemic|epidemic|infected|infection|outbreak|disease)', '', p).strip() for p in preds]
    preds = [re.sub(r'^(\(?covid[ -]?19\)?|\(?2019[ -]?ncov\)?|patients?|coronavirus(es)?|outbreak)', '', p).strip() for p in preds]#some leftover abbreviations that do not add much
    preds = [re.sub(r'^\d+ (cases|parients?)?', '', p).strip() for p in preds]#raw numbers at beginning of result do not add much
    preds = [re.sub(r'^(hospitalized patients|infected patients)( with)?', '', p).strip() for p in preds]

    preds= ["" if len(p) <= 1 else p for p in preds]#thanks to the post-processing above we sometimes have single letters. a single latter does not add much anyway so it can be deleted

    preds = [re.sub(r'^(patients?|coronavirus(es)|cases?|people)$', '', p).strip() for p in preds]
    preds = [re.sub(r'^(=-|[.,:;]|\d+ )', '', p).strip() for p in preds]


    preds = [re.sub('[!?.,:;*&%]$', '', p).strip() for p in preds]#no need for those chars
    preds = [re.sub(r'^\((covid-19|sars-cov-2)\)', '', p).strip() for p in preds]
    preds = [re.sub(r'^(outbreak|coronavirus disease|with 2019 novel coronavirus)', '', p).strip() for p in preds]

    ###end-string cleaning
    preds = [re.sub(r'(with |of )?\(?(covid-19|sars-cov-2)\)?\s?(patients?|outbreak)?$', '', p).strip() for p in preds]

    preds = [re.sub(r'^-?(confirmed|infected|infection)', '', p).strip() for p in preds]
    preds = [re.sub(r'^-=?', '', p).strip() for p in preds]#some unnecessary leftovers
    preds = [re.sub(r'^(\d+[.,])+(\s?(cases?|patients?)( of)?)', '', p).strip() for p in preds]
    preds = [re.sub(r'^(of )?novel coronavirus(\s?\((2019[-]?ncov | covid[-]?19)\))?', '', p).strip() for p in preds]


    preds = [re.sub(r'^(cases|(20)?19|patients)$', '', p).strip() for p in preds]#these single and stripped down categories dont make much sense either

    if mode=="Condition":
        #preds = [re.sub(r'^(novel coronavirus)\s?(infection|disease)?', '', p).strip() for p in preds]
        preds = [re.sub(r'^-?(confirmed|infected|infection)', '', p).strip() for p in preds]


    ###ids = [re.sub(r'(^\d+)(\..+)', r'\1', i) for i in ids]  # for pmids get abstract ids, not sentence ids
    ids = [re.sub(r'(^.+)(_.+)', r'\1', i) for i in ids]  #TODO check how to split them now

    mylist = zip(preds, ids)
    print(len(preds))
    mylist = [entry for entry in mylist if entry[0] != '']  # delete empty predictions for efficiency
    print(len(mylist))

    conditions = defaultdict(list)
    for entry in tqdm(mylist):  # add all ids to the relevant keyword
        conditions[entry[0]].append(entry[1])

    #print('length of cancer entry')
    #print(conditions['cancer'])

    condition_to_id = {}
    for k, v in conditions.items():  # make most relevant abstracts appear in fromt

        l1 = sorted(v, key=collections.Counter(v).get, reverse=True)  # get most relevant abstracts in front
        deduped = []  # deduplicate
        for item in l1:
            if item not in deduped:
                deduped.append(item)
        condition_to_id[k] = deduped

    #print('length of cancer entry')
    #print(len(condition_to_id['cancer']))
    #print(condition_to_id['cancer'])

    # make the deduplicated, alphabetically sorted dataframe
    alphabetic_sorted = sorted(Counter(preds).most_common(), key=lambda tup: tup[0])
    final_sorted = sorted(alphabetic_sorted, key=lambda tup: tup[1], reverse=True)

    if final_sorted[0][0] == "":  # if most common is an empty string: is around 85618 sentences per 10000 abstracts
        del (final_sorted[0])

    df = pd.DataFrame(final_sorted, columns=['Condition', 'Counts'])
    conditions = list(df['Condition'])

    id_column = []  # add pubmed ids
    pubmed_strings = []
    for c in conditions:
        id_column.append(condition_to_id[c])
        pubmed_strings.append(" ".join(condition_to_id[c]))

    ###append the data
    df['Pubmed_Search'] = pd.Series(pubmed_strings).values
    # df['Pubmed_ID']=pd.Series(id_column).values

    print(df.head())

    # k = [print(final) for final in final_sorted]

    return df


def connectInput(paths,mode="Condition"):


    predicted = []
    destinationDf = pd.DataFrame(columns=['ID', 'Sent'])

    try:
        for path in paths:#for each subfolder of "predictions"
            print('reading files from: {}'.format(path))
            destination = os.path.join(path, "sent_map.csv")  # file with sentences and their ids
            if mode == "Condition":
                preds = os.path.join(path, "predictions_C.json")
            elif mode == "Intervention":
                preds = os.path.join(path, "predictions_I.json")
            else:
                preds = os.path.join(path, "predictions_P.json")

            with open(preds) as feedsjson:
                my_feeds = json.load(feedsjson)

            predicted.extend(my_feeds.items())#get predictions from this batch and add them to the other predictions
            destinationDf = pd.concat([destinationDf, pd.read_csv(destination)])
            print('Opened prediction files and appended content successfully..')
    except:
        pass

    print(len(predicted))

    preds = [p[1] for p in predicted]
    ids = [p[0] for p in predicted]
    results = process_input(preds, ids,mode)

    if mode == "Condition":
        results.to_csv(
            os.path.join(os. getcwd(),'predictions', 'predictionsLENA_C.csv'))
    elif mode == "Intervention":
        results.to_csv(
            os.path.join(os. getcwd(),'predictions', 'predictionsLENA_I.csv'))
    else:
        results.to_csv(
            os.path.join(os. getcwd(),'predictions', 'predictionsLENA_P.csv'))

    # print(len(ids))
    print(len(preds))
    # print(len(destinationDf['ID']))

    # print(type(ids[0]))
    # id_orig= list(destinationDf['ID'])
    # print([item for item, count in collections.Counter(id_orig).items() if count > 1])#print duplicate
    # not_in=[]
    # for ido in tqdm(id_orig):
    # if ido not in ids:
    # not_in.append(ido)
    # print(len(not_in))
    # print(not_in)

    # destinationDf['Lena_Sensitive'] = preds
    # destinationDf.to_csv('C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Sarah P mining\\immunoglobulin\\predictions\\file_1\\predictionsLENA.csv')


def deduplicate_predictions(path, top_n=50000, mode="C"):
    #Rather resource intensive way to deduplicate and cluster. Read top mined candidate and add it to unique list and greate a dict entry as key.
    # #Then, in decreasing popularity, all other mining results ar checked. If a unique condition matches a subset of the checked condition then the checked condition is clustered
    # with the unique condition as antry in its dict. Otherwise, a new unique condition is created, which means that towards the end each new mining result might go through 100s of checks for clustering..
    #Probably this can be done way more efficiently!!!!!!!
    ##
    #
    df = pd.read_csv(path)
    print("Number of entries in df before deduplication: {}".format(df.shape[0]))
    condition = []
    counts = []
    Pubmed_Search = []
    included_conditions = {}

    unique_conditions = 0
    with tqdm(total=df.shape[0]) as pbar:
        for index, row in df.iterrows():  # whole df
            found = False
            # entry= row[1].replace("(", "\(").replace(")", "\)")
            entry = row[1]
            for c in condition:  # our deduplicated conditions

                if re.search(r"\b{}\b".format(re.escape(c)),
                             entry):  # escape any parentheses etc that could be mistaken for regex metacharacters
                    found = True
                    included_conditions[c].append(entry)

                    break

            if found == False and unique_conditions <= top_n:  # no prev entry, need to create one

                condition.append(entry)
                included_conditions[entry] = []
                unique_conditions += 1
            pbar.update(1)
    pbar.close()
    k = included_conditions.keys()
    v = []
    for val in included_conditions.values():
        all_v = ";; ".join(va for va in val)
        v.append(all_v)

    deduped = pd.DataFrame(zip(k, v))
    if mode == "C":
        deduped.to_csv(
            os.path.join(os. getcwd(),'predictions', 'C_deduped.csv'))

    elif mode == "I":
        deduped.to_csv(
            os.path.join(os. getcwd(),'predictions', 'I_deduped.csv'))
    else:
        deduped.to_csv(
            os.path.join(os. getcwd(),'predictions', 'P_deduped.csv'))

    print(len(condition))


def custom_deduplication(path):
    # create a map with a list of custom conditions and search patterns. If one wants to focus on special conditions, rather than on the top n
    #of course it would be better to auto-assign mesh terms to the mining results, then one could query and visualise the dataset easier.
    # #Maybe someone with an implemented Mesh ontology/assignment algorihm?
    custom_conditions = [

        (r'(disseminat|demyelinat|post\w+)(ed|ing)? encephal(omye|iti)', "Acute disseminated encephalomyelitis "),
        (r'\bmyelitis|myelopathy syndrome', "Transverse myelitis"),
        (r'(neuromyelitis optica)|(devic(s|\'s)?\b)|(\bnmo\b)', "Neuromyelitis optica"),
        (r"(autoimmun|anti[- ]?)(.+)? encephal|(VGKC|potassium channel encephal)", "Autoimmune encephalitis "),
        (r'fac(e|ial)[- ]pain', 'Chronic facial pain'),
        (r'regional pain syndr|\bcrps\b', 'Chronic regional pain syndrome (CRPS)'),
        (r'(childhood|refractor|resist)(.+)? epilep', 'Intractable childhood epilepsy'),
        # (r'(anti[- ]?n(.+)|auto[- ]?immune) encephal|(\bvgkc), 'Autoimmune encephalitis'),
        (r'(atopic|disseminat|infant)(.+)((neuro)?derma|eczem)', 'Atopic dermatitis/eczema'),
        (r'(anti[- ]?phospho|hugh|asher).+syndro', 'Antiphospholipid syndrome')
    ]

    df = pd.read_csv(path)
    print("Number of entries in df before deduplication: {}".format(df.shape[0]))
    # condition = []
    counts = []
    Pubmed_Search = []
    included_conditions = {i[1]: [] for i in custom_conditions}  # add second entry of tuple as key in dict

    counter = 0
    with tqdm(total=df.shape[0]) as pbar:
        for index, row in df.iterrows():  # whole df
            found = False
            # entry= row[1].replace("(", "\(").replace(")", "\)")
            entry = row[1]
            for c in custom_conditions:  # our deduplicated conditions

                if re.search(c[0], entry):  # escape any parentheses etc that could be mistaken for regex metacharacters
                    # if c[1]=='Autoimmune encephalitis':
                    # print(entry)
                    # counter += 1
                    included_conditions[c[1]].append(entry)  # append entry under its topic, as defined above

                    break

            pbar.update(1)
    pbar.close()
    print("found {} entities".format(counter))
    k = included_conditions.keys()
    v = []
    for val in included_conditions.values():
        all_v = ";; ".join(va for va in val)
        v.append(all_v)

    deduped = pd.DataFrame(zip(k, v))
    deduped.to_csv(
        "P_map_grey.csv")

def rewrite_cord_data(path, max_rows = 10000000, min_date=2020):
    #
    #Function takes the metadata with mag mapping file and reformats it a bit, changing column names
    #

    df = pd.read_csv(path)#get data
    #reassign column names to fit my existing scripts
    # #cord_uid becomes pmid, title abstact get capital beginning
    df.columns = [
    "PMID",
    "sha",
    "source_x",
    "Title",
    "doi",
    "pmcid",
    "pubmed_id",
    "license",
    "Abstract",
    "publish_time",
    "authors",
    "journal",
    "Microsoft Academic Paper ID",
    "WHO hash Covidence",
    "has_full_text",
    "full_text_file",
    "url"]

    new_time=[]
    for time in df["publish_time"].values:

        t= str(time)
        t= t[:4]
        #print(t)
        try:
            new_time.append(int(t))
        except:
            #print("no data found, value is: {}".format(t))
            new_time.append(0)
        #new_time.append(int(str(time)[-4:]))
    df["publish_time"]  =  new_time

    df=df[df["publish_time"] >= min_date]
    df = df[:max_rows]#keep all, or keep only first n
    df.to_csv("covid_data.csv")
    print("SAved reformatted input data as covid_data.csv in working directory.")

###################
# #first, we need to bring these data in a slightly new format (just changing some column names here to fit my original scripts)

#rewrite_cord_data("metadata.csv", max_rows=150000, min_date=2020)
######################

###########
#  now make the QA input. paths are given as lists, in case one has many smaller input files (my original data came in spreadsheets of 10.000 pubmed abstract each)
#makeInput(["covid_data.csv"], types="I")#types P and C and I
########################

#############
#do something with the preddictions: first we need to post-processs them a tiny bit, then link thm to the original dataset and sort them by frequency
connectInput(["predictions//files_1"], mode="Intervention")   #modes "Condition" or else for Patints  #to make unified csv file

##################################
#The we runa simple unsupervised clustering based on substrings in the mined data

deduplicate_predictions("predictions\\predictionsLENA_I.csv", mode="I")

##############################
##if one is interested in certain conditions, or happy to implement some linking with MeSH or ontologies then this is a good starting point here. Right now it works with regexes
####custom_deduplication("C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Sarah P mining\\immunoglobulin\\predictions\\P_copy.csv")