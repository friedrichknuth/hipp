import cv2
import concurrent
import os
import pathlib
import numpy as np
import pandas as pd
import psutil
import urllib

import hipp


"""
Library query and download historical image data from public archives.
"""

def download_image(output_directory, 
                   payload):
    url, output_file_base_name = payload
    output_file = os.path.join(output_directory, output_file_base_name+'.tif')
    urllib.request.urlretrieve(url,output_file)
    final_output = hipp.utils.optimize_geotif(output_file)
    os.remove(output_file)
    os.rename(final_output, output_file)
    return output_file
    
def EE_login(username,
             password,
             m2mhost = 'https://m2m.cr.usgs.gov/api/api/json/stable/'):

    data = {'username' : username, 
            'password' : password}
    
    url = m2mhost + 'login'
    api_key = ee_sendRequest(url, data)
    
    return api_key
    
def EE_sendRequest(url, data, apiKey = None):  
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers = headers)    
    
    try:
        httpStatusCode = response.status_code
        if response == None:
            print("No output from service")
            sys.exit()
        output = json.loads(response.text)
        if output['errorCode'] != None:
            print(output['errorCode'], "- ", output['errorMessage'])
            sys.exit()
        if  httpStatusCode == 404:
            print("404 Not Found")
            sys.exit()
        elif httpStatusCode == 401: 
            print("401 Unauthorized")
            sys.exit()
        elif httpStatusCode == 400:
            print("Error Code", httpStatusCode)
            sys.exit()
    except Exception as e:
        response.close()
        print(e)
        sys.exit()
    response.close()
    
    return output['data']

def NAGAP_download_images_to_disk(image_metadata, 
                                  output_directory='input_data/raw_images',
                                  file_name_column='fileName',
                                  image_type_colum='pid_tiff',
                                  base_url = 'https://arcticdata.io/metacat/d1/mn/v2/object/',
                                  verbose=True):
                            
    print("Downloading images...")
    p = pathlib.Path(output_directory)
    p.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(image_metadata, type(pd.DataFrame())):
        df = pd.read_csv(image_metadata)
    else:
        df = image_metadata

    df['urls'] = base_url + df[image_type_colum]
    urls, filenames = df['urls'], df[file_name_column]
    
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    future_to_url = {pool.submit(hipp.dataquery.download_image,
                                 output_directory,
                                 x): x for x in zip(urls, filenames)}
    results=[]
    for future in concurrent.futures.as_completed(future_to_url):
        r = future.result()
        results.append(r)
        print(r)
        
    return output_directory

def NAGAP_pre_select_images(nagap_metadata_csv,
                            bounds=None,
                            roll=None,
                            year=None,
                            month=None,
                            day=None,
                            output_directory=None,
                            verbose=True):
    """
    bounds = (ULLON, ULLAT, LRLON, LRLAT)
    year   = 77 # e.g. for year 1977
    """
    print("Selecting images based on:")           
    df = pd.read_csv(nagap_metadata_csv, dtype=object)
    df['Longitude'] = df['Longitude'].astype(float)
    df['Latitude'] = df['Latitude'].astype(float)
    
    if not isinstance(bounds,type(None)):
        print('bounds:', bounds)
        df = df[(df['Longitude']>bounds[0]) & 
                (df['Longitude']<bounds[2]) & 
                (df['Latitude']>bounds[3]) & 
                (df['Latitude']<bounds[1])]
    
    if not isinstance(roll,type(None)):
        print('roll:', roll)
        df = df[df['Roll'] == roll]
        
    if not isinstance(year,type(None)):
        print('year:', year)
        df = df[df['Year'] == str(year)]
        
    if not isinstance(month,type(None)):
        print('month:', month)
        df = df[df['Month'] == str(month)]
        
    if not isinstance(day,type(None)):
        print('day:', day)
        df = df[df['Day'] == str(day)]
        
    df = df.reset_index(drop=True)
    
    if len(list(set(df['Roll'].values))) > 1:
        print('NOTE: Results contain multiple camera rolls:')
        for i in list(set(df['Roll'].values)):
            print(i)
        
    if len(list(set(df['Year'].values))) > 1:
        print('NOTE: Results contain multiple years:')
        for i in list(set(df['Year'].values)):
            print(i)
            
    if len(list(set(df['Year'].values))) == 1 and len(list(set(df['Month'].values))) > 1:
        print('NOTE: Results contain multiple months:')
        for i in list(set(df['Month'].values)):
            print(i)
        
    if not isinstance(output_directory,type(None)):
        out = os.path.join(output_directory,'targets.csv')
        df.to_csv(out, index=False)
        return out
    
    else:
        return df
        
