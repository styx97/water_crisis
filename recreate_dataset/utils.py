import re 
import json
import pandas as pd
import emoji
import string 
from pprint import pprint
import random
from collections import Counter 

"""
Remove full and partial emojis
"""
# code from https://stackoverflow.com/a/51785357/6684726 License : https://creativecommons.org/licenses/by-sa/4.0/w

try:
    uchr = unichr  
    import sys
    if sys.maxunicode == 0xffff:
        # narrow build, define alternative unichr encoding to surrogate pairs
        # as unichr(sys.maxunicode + 1) fails.
        def uchr(codepoint):
            return (
                unichr(codepoint) if codepoint <= sys.maxunicode else
                unichr(codepoint - 0x010000 >> 10 | 0xD800) +
                unichr(codepoint & 0x3FF | 0xDC00)
            )
except NameError:
    uchr = chr  # Python 3

# Unicode 11.0 Emoji Component map (deemed safe to remove)
_removable_emoji_components = (
    (0x20E3, 0xFE0F),             # combining enclosing keycap, VARIATION SELECTOR-16
    range(0x1F1E6, 0x1F1FF + 1),  # regional indicator symbol letter a..regional indicator symbol letter z
    range(0x1F3FB, 0x1F3FF + 1),  # light skin tone..dark skin tone
    range(0x1F9B0, 0x1F9B3 + 1),  # red-haired..white-haired
    range(0xE0020, 0xE007F + 1),  # tag space..cancel tag
)
emoji_components = re.compile(u'({})'.format(u'|'.join([
    re.escape(uchr(c)) for r in _removable_emoji_components for c in r])),
    flags=re.UNICODE)

def remove_emoji(text, remove_components=False):
    cleaned = emoji.get_emoji_regexp().sub(u'', text)
    if remove_components:
        cleaned = emoji_components.sub(u'', cleaned)
    return cleaned


def remove_punc(text):
    exclude = set(string.punctuation)
    exclude = exclude - set([".","!","?"])
    s1 = ''.join(ch for ch in text if ch not in exclude)
    return s1.lower()

def shrink_delimiters(text):
    # replace occurence of a char from the set (!,?,.) after a char from the same set
    flag = 0 
    for char in [".","?","!"] : 
        if char in text:
            flag = 1
    
    if flag:
        
        text1 = re.sub(r'[\.\?\!]+(?=[\.\?\!])', '', text.strip())
        text1 = " ".join(text1.strip().split())
        text1 = re.split(r'[.?!]',text1)
        return [x.strip() for x in text1]

    else :
        return [text]

def comments_to_chains(sentence_list):
    '''
    input format : [[sentence1, attribution, topic_set],
                    [sentence2, attribution, topic_set],
                    ...
                    ]
    output format : [[chain1, attribution, topic],
                     [chain2, attribution, topic],
                     ...
                     ]
    
    '''
    
    total_chains = []
    for s in range(len(sentence_list)): 
        
        
        if sentence_list[s][1] == 0: 
            
            # attribution is zero 
            total_chains.append([sentence_list[s][0], 0, "no_attribution"])
        
        else : 
            
            if len(sentence_list[s][2]) > 0 : 
                #for topic_head in sentence_list[s][2]:
                
                while len(sentence_list[s][2]) != 0 : 
                    topic_head = sentence_list[s][2][0]
                    #print("Current Topic Head",topic_head)
                    #print(sentence_list)
                    
                    
                    # there is only 1 chain for one topic head : 
                    chain = []
                    
                    # enter the topic head and the corresponding sentence - 
                    chain.append(sentence_list[s][0])
                    #print(topic_head, chain)
                    # remove the topic from the topic heads
                    sentence_list[s][2].remove(topic_head)
                    #print(chain)
                    
                    if not s == len(sentence_list) - 1: 
                        
                        flag = True
                        for ns in range(s+1, len(sentence_list)):
                            if not topic_head in sentence_list[ns][2]:
                                # The chain is broken , add it to list of chains 
                                if flag: 
                                    total_chains.append([" ".join(chain),1,topic_head])
                                    flag = False
                                    break 

                            else : 
                                # add sentence to chain and remove topic : 
                                chain.append(sentence_list[ns][0])
                                sentence_list[ns][2].remove(topic_head)
                               
                        if flag :
                            total_chains.append([" ".join(chain),1,topic_head])
                        
                    else : 
                        total_chains.append([" ".join(chain),1,topic_head])
                        
    
                    
    return total_chains

def make_comments(dataframe):
    
    comments = set(dataframe.comment_id)
    final_chains = []
    
    for comment_id in comments: 
        temp_info = []
        temp_chains = []
        df_temp = dataframe.loc[dataframe['comment_id'].isin([comment_id])]
        
        for index, row in df_temp.iterrows():
            temp_topics = [row.primary_attribution, row.secondary_attribution, row.multiple_attribution_1, row.other_attribution]
            
            if isinstance(row.other_attribution, str):
                t = row.other_attribution.split(',')
                for elem in t: 
                    #print(elem)
                    temp_topics.append(elem.strip()) 
            #print(temp_topics)   
                
            temp_topics = [x.strip() for x in temp_topics if not pd.isnull(x) and not x in ['other','others','population','infrastructure']]
#             if ['other', np.nan] in temp_topics:
#                 temp_topics.remove('other')
            #print(temp_topics)
            if not pd.isnull(row.sentence) : 
                temp_info.append([row.sentence, row.attribution, temp_topics])
       
        #print(f"{temp_info}\n")
        temp_chains = comments_to_chains(temp_info)
        
        for chain in temp_chains:
            final_chains.append(chain)
            
    
    return final_chains    

def replace_no_attrib(dataframe, topics_sets):
    
    attribs = dataframe.attrib_words 
    count = Counter(attribs)
    # print("Number of chains with no attribution is {}".format(count['no_attribution']))
    # print("Number of chains with attribution  is {}".format(dataframe.shape[0] - count['no_attribution']))
    counter = 0 
    random.seed = 2020
    all_attributions = list(set(dataframe.attrib_words) - set(["no_attribution"]))
    #print(all_attributions)
    new_row = []
    
    for index, row in dataframe.iterrows():
        
        if row.attrib_words == "no_attribution":
            counter += 1 
            #print(counter)
            assigned_topic = random.choice(list(set(all_attributions) - topics_sets[row.sentences]))
            #print(assigned_topic)
            new_row.append(assigned_topic)
            
        else :   
            
            new_row.append(row.attrib_words)
    
    #assert(len(all_attributions) == len(dataframe.attrib_words))
    dataframe = dataframe.assign(attrib_words = pd.Series(new_row))
    
    return dataframe
    