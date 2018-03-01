#!/usr/bin/env python

import datetime as dt
import json, sys
from googleapiclient.discovery import build


def searchMatchCount(search_term):
    # Keys from Google CustomSearch
    search_engine_id = '000883281515735056312:ddijpupyziw'
    api_key = 'AIzaSyDiSNsnGi4AAEoRsKrPWjj2I_9uWJaiJk8'
    
    service = build('customsearch', 'v1', developerKey=api_key)
    collection = service.cse()

    match_count = 0
    num_requests = 10
    for i in range(0, num_requests):
        start_val = 1 + (i * 10)
        request = collection.list(q=search_term,
            num=10, 
            start=start_val,
            cx=search_engine_id
        )
        response = request.execute()
        for item in response['items']:
            match_count += sum(item['title'].count(x) for x in search_term.split())
            match_count += sum(item['snippet'].count(x) for x in search_term.split())
    return match_count

def searchTotalCount(search_term):
    return 0

def searchEngineTest(search_term):
    match_count = searchMatchCount(search_term)
    total_count = searchTotalCount(search_term)
    #employ log magicks
    final_score = match_count
    return final_score

if __name__ == '__main__':
    print(str(searchEngineTest("effective support")))
