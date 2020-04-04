import json
import pandas as pd
import re
from glob import glob
import random
import os
from random import shuffle
import codecs 
from tqdm import tqdm

def span2text(start, end, idDict, context):
    #gives answer text based on spans
                    ans = [idDict[ind] for ind in range(start, end+1)]
                    cont=context.split()
                    ansSpan=0
                    actual_text=''
                    sents=[]
                    for ind, text in enumerate(cont):
                        #print(text + ' ' + str(ans))
                        if ans[0] == text:
#                            print('---------')
                            
                            
                            ending=ind + len(ans)-1
                            try:
                                if (cont[ending]) == ans[-1]:
                                    
#                                    ansSpan = context.index(actual_text)
                                    actual_text=' '.join(ans)
                                    index_start=context.index(actual_text)#test, to see if index is found
                                    break
                                    #print(ansSpan)
                            except:##########this happens if answers were annotated across sentence boundaries. e.g. 'Error with index 8 in ['Iodixanol', 'caused', 'significantly', 'less', 'discomfort', 'than', 'iohexol', '.']: word iohexol span is this long: 3. This is the ans: ['iohexol', '.', 'Iodixanol']'
                                
                                answer=' '.join(ans)##the given answer. might exceed sentence boundaries
                                
                                
                                parts= answer.split('.')#sine sometimes annotations are across sentences
                                sents = [sent.strip() for sent in parts]
                                
                                for candidate in sents:
                                    if candidate in context and candidate !='':
#                                        ansSpan = context.index(candidate.strip())
                                        actual_text = candidate.strip()
                                        
                                        #print('found {} ---- {}'.format(actual_text,context))
                                        break
                                if actual_text == '':
                                    print('not found {} ----\n {}'.format(sents,context))
                                
                    try:
                        return(actual_text, context.index(actual_text))
                    except:
                        
                        print('not found {} ----\n {}'.format(actual_text,context))#should not happen, since annotated data files seem to exist reliably
                        return(actual_text, 0)
                     
           





    
    
def addAbsData(spanFile, tokFile,entity, makeTest, starting_spans=True, spanID=1, undersample_frac = 0.3):

    #spanFile, tokFile are given automatically, they represent 1 abstract with its tokens and labels
    #starting_spans: if the script should use the binary annotations, or the herarchical
    #spanID: the id tht is considered as the positive case. 1 in case of starting_spans, and 1 to x inc ase of the herarchical labels
    #undersample_frac: this many percent of sentences will be randomly deleted if they dont contain a positive example, e.g. 30% if the fraction is 0.3. Deletions only happen in training data, not in testing data!
    

    if starting_spans:#means that we lookat the file that has starting spans (only abels 0 and 1)
        if entity == "P":
            quests=['Who was treated?','What were the criteria for enrolment?','What were the inclusion criteria?','Who was enrolled in the study?','What were participants diagnosed with?', 'Which participants took part in the study?']
        elif entity == "I":
            quests = ['Which intervention did the patients receive?',
                      'Which intervention did the participants receive?', 'What was the intervention?',
                      'What did the patients receive?', 'What did the participants receive?',
                      'What was the intervention given to the participants?']

        elif entity == "O":
            quests = ['Which outcomes were measured?', 'What did the study measure?', 'Which endpoints were measured?',
                      'What were the measured outcomes?', 'What were the measured endpoints?',
                      'What were the primary or secondary endpoints?', 'What were the primary or secondary outcomes?']

    else:#get the hierarchical labels
        if spanID==4 and entity == "C":#
            quests = ['Which condition did the participants have?', 'Patients with which medical conditions were included?', 'What was the medical condition?', 'What was the condition?']
        
        ##gender is 2
        if spanID==2:
            quests = ['What sex were the participants?', 'What was the patient\'s sex', 'What gender were the participants?', 'What was the patient\'s gender', 'Were there male or female participants included?']
        
        if spanID==1:
            quests = ['What age were the participants?', 'How old were the participants?', 'What age were the patients?', 'How old were the patients?', 'What was the age in the population?']
        
        if spanID==3:
            quests = ['What was the sample size?','How big was the sample size?','How big was the population','What was the size of the population?', 'How many participants were enrolled?', 'How many partients took part in the study?', 'How many participants took part in the trial?', 'How many partients were enrolled?']
        
    quest = random.sample(quests, k=1)[0]#as this returns list, we take first item of list###random question
    
    
    someI = codecs.open(spanFile, encoding='utf-8').read().split()
    someTok=codecs.open(tokFile, encoding='utf-8').read().split()
    
    #print("-------------------------")print all labels and tokens for debugging
    #print(someI)
    #print("Tokens:")
    #print(someTok)
    
    
    topicString = os.path.splitext(os.path.basename(spanFile))[0].split('.')[0]#pmid as topic
    
    contexts=[]#sent with word index tuples
    text=''#one sent
    
    id_to_sent = {}
    id_to_word = {}
    
    counter=0
    
    end_ids=[]
    for index, word in enumerate(someTok):
        
        #######indexes of start spans are one too early
        text = '{} {}'.format(text,word).strip() # restore full sentences, as bert takes whole context sentences and does splitting itself
        id_to_sent[index]=counter#set sentence number for this id    
        id_to_word[index]=word
                
        
        if re.search('^[.!?]$', word):#regex anchors to filter single fullstop or other common end-of-sentence punctuation
            
            contexts.append(text)#get ready for next sent
            text=''
            counter+=1#next sentence
            end_ids.append(index)
            
    domainDict={'title':topicString,'paragraphs':[]}##append many, change this to constructor

       
    starts=[]
    ends=[]
    sents=[]
    
    for ind, value in enumerate(someI):###get start and end span indexes for this pico
        #print(str(ind) + ' ' + str(value) +' ' + someTok[ind])
        
        if int(value) == spanID:#value is string because it is read from txt file
            
            if ind ==0 or int(someI[ind-1]) != spanID:#if we are at start or if previous is differentt, then we have a beginning span
                starts.append(ind)
                
                #print('start appended at ' + str(ind))
            if len(someI)-1 == ind or int(someI[ind+1]) != spanID: #opposite case to above
                ends.append(ind)
                sents.append(id_to_sent[ind])#get sentence number for that span
                #print('end appended at ' + str(ind))
                #print('---')
                
              
    spans = list(zip(starts, ends, sents))#print all identified spans for debugging
    #print('spans:')
    #print(spans)
    
    
    for context in contexts:
        domainDict['paragraphs'].append({'qas':[], 'context':context})
    
             
                    
    to_delete = []
    
    for index, paragraph in enumerate(domainDict['paragraphs']):
        qaDict={'question':quest, 'id':'{}{}'.format(random.random(), random.random()), 'answers':[], 'is_impossible':False} 
        
        
        for span in spans:
            
            if span[2] == index:#if sentence appears in the span data
                
                txt, spanStart= span2text(span[0], span[1], id_to_word, paragraph['context']) 
                
                if makeTest:#append list of posible answers
                    qaDict['answers'].append({'text': txt, 'answer_start': spanStart})  ### multiple answers
                else:#at training, only one anser can be given per span
                    qaDict['answers']=[{'text': txt, 'answer_start': spanStart}]
#                print('---')

                
        if len(qaDict['answers']) == 0:#no answer was found
            qaDict['is_impossible'] = True
            qaDict['plausible_answers']= [
                {
                  "text": ' '.join(paragraph['context'].split(' ')[:2]),#first 2 words
                  "answer_start": 0
                }
              ]
            
            if random.random() < undersample_frac and makeTest == False:#undersampling, but only when we are not in test data mode. test data should represent the reality
                to_delete.append(index)
                
          
        paragraph['qas'].append(qaDict)
        
        
        
        #print(len(paragraph['quas']))     
        #print(qaDict['answers'])

    for index in sorted(to_delete, reverse=True):#do the actual undersampling by deleting the randomly selected sentences that had no answer
        del domainDict['paragraphs'][index]
    #print(domainDict)
    return domainDict

def make_data(entity="C", makeTest=True, undersample_frac = 0.3, add_dev=0,add_train=50):
    #
    #Function to create train or test data in form of a JSON file
    #
    #param: makeTest; if True the script creates test data (no undrsampling, no aditional SQuAD domains.)
    #
    #param: undersample_frac: if one want to do some undersampling then the fraction can be given here. This many percent of sentences will be randomly deleted if they dont contain a positive example, e.g. 30% if the fraction is 0.3. Deletions only happen in training data, not in testing data!
    #
    #add_dev: number of original squad domains that shall be added to our evaluation file.. ) per default, although one can add some extra domains to see if hte model retains general question-answering capabilities
    #dd_train: number of original squad domains that shall be added to our training file! This is where one decides how much extra data to use :)
    ################

    path_to_SQUAD_dev = 'C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\data\\squad\\dev-v2.0.json'
    path_to_SQUAD_train= 'C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\data\\squad\\train-v2.0.json'

    if makeTest:
        print('Reading test documents...')
        if entity == "I":
            span_fnames = glob("C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\interventions\\test\\gold\\*")######for I
        elif entity == "O":
            span_fnames = glob("C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\outcomes\\test\\gold\\*")###for O
        elif entity == "P":
            span_fnames = glob("C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\participants\\test\\gold\\*")###for P

        elif entity=="C":#test data for conditions, which is a hierarchical label under the P entity
            span_fnames = glob(
                "C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\hierarchical_labels\\participants\\test\\gold\\*")  ######for P extra spans
        else:
            print("Error, The paths for this entity are not yet defined")



    else:  # train data
        print('Reading train documents...')
        if entity == "I":
            span_fnames = glob("C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\interventions\\train\\*")##FOR I
        elif entity == "P":
            span_fnames = glob(
                "C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\participants\\train\\*")
        elif entity == "O":
            span_fnames = glob("O:\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\outcomes\\train\\*")

        elif entity == "C":#train data for conditions, which is a hierarchical label under the P entity
            span_fnames = glob("C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\annotations\\aggregated\\hierarchical_labels\\participants\\train\\*")######for P extra spans
        else:
            print("Error, The paths for this entity are not yet defined")


     #reading all token files
    print('Reading token files...')
    toks = glob(
        "C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\documents\\*.tokens")

    tok_fnames = []
    base = "C:\\Users\\xf18155\\OneDrive - University of Bristol\\MyFiles-Migrated\\Documents\\Python Scripts\\pytorch\\ebm_nlp_2_00\\documents\\"

    for fname in span_fnames:
        pmid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]  # get pubmed id, code borrowed from nye github repo
        tok_fnames.append(os.path.join(base, str(pmid) + '.tokens'))

    inputs = list(zip(span_fnames, tok_fnames))  # list of corresponding tuples of source file and span file
    #########################get files
    versionString="v2.0"
    data = {'version': versionString, 'data': []}#to have possibility of pure new squad training data

    print('Converting entity annotations to SQuAD format...')
    for inp in tqdm(inputs):
    #for inp in inputs:
    #    print(inp)
        spf=inp[0]#annotated span comes first in the data
        tkf=inp[1]


        if entity =="C":#we just need to give a few more params for the hierarchical labels, so that the script knows which label to use; C has the ID 4 in the population sub-classes
            domainDict = addAbsData(spf, tkf, entity, makeTest, starting_spans=False, spanID=4, undersample_frac = undersample_frac)#this has an automatic 0,3 undersampling, needs to be changed by undersample_frac param
        else:#the main clsses (PIO dont need the extra info)
            domainDict = addAbsData(spf, tkf, entity, makeTest, undersample_frac = undersample_frac)#this has an automatic 0,3 undersampling, needs to be changed by undersample_frac param

        data['data'].append(domainDict) #append this domain dict and make the next :)


    if makeTest:
        with open(path_to_SQUAD_dev, encoding='utf-8') as feedsjson:#path to original squad dev file - if you ant to mix some original dev data
            feeds = json.load(feedsjson)
            print("Number of domains available in original SQuAD dataset:")
            print(len(feeds['data']))
            try:
                print('Adding {} extra SQuAD domains and saving JSON'.format(add_dev))
                feeds['data']= random.sample(population=feeds['data'],k=add_dev)#mixing data, no need to shuffle for dev
            except:
                pass


            feeds['data'] =feeds['data']+ data['data']
            print("Final number of domains in data:")
            print(len(feeds['data']))
            #shuffle(feeds['data'])  ###no need to shuffle the eval set
            with open('dev-v2.0_'+entity +'.json', mode='w') as f:
                f.write(json.dumps(feeds, indent=2))
    else:

        with open(path_to_SQUAD_train) as feedsjson:#path to original train file
            feeds = json.load(feedsjson)

            print("Number of domains available in original SQuAD dataset:")
            print(len(feeds['data']))

            try:
                print('Adding and shuffeling {} extra SQuAD domains and saving JSON'.format(add_train))
                print(len(feeds['data']))
                feeds['data']= random.sample(population=feeds['data'], k=add_train)#add N domains
            except:
                pass


            feeds['data'] =feeds['data']+ data['data']
            print("Final number of domains in data:")
            print(len(feeds['data']))
            shuffle(feeds['data'])  #shuffle to avoid having small/big gradients only with same data
            ############training data
            with open('train-v2.0_'+entity +'.json', mode='w') as f:
                f.write(json.dumps(feeds, indent=2))
        
make_data(entity="C", makeTest=True, undersample_frac = 0.3, add_dev=0,add_train=20)
    



    