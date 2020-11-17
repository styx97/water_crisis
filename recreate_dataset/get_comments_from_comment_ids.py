import json
import os
import requests
import sys
import pandas as pd
from pprint import pprint
from json.decoder import JSONDecodeError
import datetime


def get_data(data_filepath): 
    """
    Get keys in a list 
    """

    data = [] 
    with open(data_filepath) as k:
        for line in k:
            data.append(line.rstrip("\n"))
    
    return data



#sys.stdout = open('/dev/stdout', 'w')

if __name__ == '__main__':
    """
    Usage : 

    
    """

    
    #API_URL = 'https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={CHANNEL_ID}&maxResults=50&order=date&pageToken={NEXT_PAGE_TOKEN}&publishedAfter={AFTER}&publishedBefore={BEFORE}&type=video&prettyPrint=true&key={KEY}'
    API_URL = 'https://youtube.googleapis.com/youtube/v3/comments?part=snippet&part=id&id={COMMENT_ID}&key={KEY}'
    keys_filepath = sys.argv[1]   
    comment_ids_path = sys.argv[2]
    save_filepath = sys.argv[3] 
    
    comment_ids = get_data(comment_ids_path)
    KEYS = get_data(keys_filepath)
    
    counter = 0 
    
    currKeyIdx = 0
    # iterating on the videos - 
    # main loop 
    for comment_id in comment_ids:
        
        #possible_dupe = glob.glob(f"fox_comments/{videoId}")
        if os.path.exists(f"{save_filepath}/{comment_id}"):
            print("This video has been iterated over")
            continue 
        else : 
            pass 
        
        count = 0 
        flag = False

        while not flag : 
            
            try : 
                next_url = API_URL.format(KEY=KEYS[currKeyIdx], COMMENT_ID=comment_id)
                print(next_url)
                
            except IndexError : 
                print("Keys exhausted")
                # print("Saving ranges having videos more than 500")
                # with open("abnormal_ranges", "w") as fp : 
                #     fp.write(f"{time_range}\n")
                sys.exit(0)
            
            try:
                req = requests.get(next_url)
                print(req.status_code)
                print('Downloading', comment_id, "Count:", counter)
                if req.status_code == 403 and json.loads(req.text)['error']['errors'][0]['reason'] == 'commentsDisabled':
                    break

                if req.status_code == 404:
                    break
                
                if req.status_code == 200 : 
                    flag = True
                    counter += 1 

                req.raise_for_status()

                cmts = req.text
                this_pg_payload = json.loads(cmts)
                present = len(this_pg_payload["items"])

                if present > 0 : 
                    raw_text = this_pg_payload["items"][0]["snippet"]["textOriginal"]
                    print(raw_text)
                    #prettified = json.dumps(this_pg_payload, indent=4)
                    
                    doc = {
                        comment_id : raw_text
                    }

                else : 
                    doc = {
                        comment_id : "Comment has been removed by the user. To consruct full dataset, contact authors."
                    }

                if not os.path.exists(save_filepath):
                    os.makedirs(save_filepath)
                        
                cur_dest = os.path.join(save_filepath,comment_id)   

                with open(cur_dest, 'w') as handle:
                    handle.write(json.dumps((doc), indent=4))
                    handle.write("\n")

                
            except Exception as e:
                
                # advance the keys a bit
                currKeyIdx += 1
                print("advancing keys")
                
                # exit if all the api keys are exhausted
                if currKeyIdx >= len(KEYS):
                    print("Ran out of keys")
                    # print("Saving ranges having videos more than 500")
                    # with open("abnormal_ranges", "w") as fp : 
                    #     fp.write(f"{time_range}\n")

                    sys.exit(0)
            