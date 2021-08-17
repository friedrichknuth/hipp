import cv2
import concurrent
import glob
import json
import os
import pathlib
import numpy as np
import pandas as pd
import psutil
import requests
import sys
import time
import urllib

import hipp


"""
Library query and download historical image data from public archives.
"""

### GENERIC FUNCTIONS

def download_image(output_directory, 
                   payload,
                   default_img_ext = '.tif'):
    url, file_name = payload
    path_name, base_name, ext = hipp.io.split_file(os.path.abspath(file_name))
    if ext == '':
        output_file = os.path.join(output_directory, base_name + default_img_ext)
    else:
        output_file = os.path.join(output_directory, base_name + ext)
    urllib.request.urlretrieve(url,output_file)
    return output_file
    
def thread_downloads(output_directory, urls, file_names):
    
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    future_to_url = {pool.submit(hipp.dataquery.download_image,
                                 output_directory,
                                 x): x for x in zip(urls, file_names)}
    results=[]
    for future in concurrent.futures.as_completed(future_to_url):
        r = future.result()
        results.append(r)
        print('Download complete for:',r)

## EARTH EXPLORER
## Using camel case to keep with API convention, where relevant, to help with debugging.

def EE_checkCompleted(entityIds,
                      output_directory):
    completed = sorted(glob.glob(os.path.join(output_directory,'*')))
    
    diff = []
    for i in entityIds:
        if any(i in s for s in completed):
            pass
        else:
            diff.append(i)
    return diff

def EE_downloadImages(apiKey,
                      entityIds,
                      label                                = 'test_download',
                      output_directory                     = 'input_data',
                      images_directory_suffix              = 'raw_images',
                      calibration_reports_directory_suffix = 'calibration_reports'):
    
    urls, file_names = hipp.dataquery.EE_stageForDownload(apiKey,
                                                          entityIds,
                                                          label = label)
                                                         
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    hipp.dataquery.thread_downloads(output_directory, 
                                    urls, 
                                    file_names)
                                    
    hipp.io.gunzip_dir(output_directory)
#     hipp.utils.optimize_geotifs(output_directory)
    
    images_directory              = os.path.join(output_directory, images_directory_suffix)
    calibration_reports_directory = os.path.join(output_directory, calibration_reports_directory_suffix)
    
    hipp.io.move_files(output_directory, images_directory, '.tif')
    hipp.io.move_files(output_directory, calibration_reports_directory, '.pdf')
    
    print('Images in:', images_directory)
    print('Calibration reports in:', calibration_reports_directory)
    
    return images_directory, calibration_reports_directory
    
def EE_filterSceneRecords(scenes):
    # Reference: https://lta.cr.usgs.gov/DD/aerial_single_frame.html
    entityIds = []
    agencies = []
    projects = []
    rolls = []
    frames = []
    recordingTechniques = []
    acquisitionDates = []
    hi_res_Available = []
    imageTypes = []
    qualities = []
    altitudesFeet = []
    imageIds = []
    focalLengths = []
    centerLats = []
    centerLons = []
    NWlats = []
    NWlons = []
    NElats = []
    NElons = []
    SElats = []
    SElons = []
    SWlats = []
    SWlons = []
    for scene in scenes:
        for entry in scene['metadata']:
            if entry['fieldName'] == 'Entity  ID':
                entityIds.append(entry['value'])
            if entry['fieldName'] == 'Agency':
                agencies.append(entry['value'])
            if entry['fieldName'] == 'Project':
                projects.append(entry['value'])
            if entry['fieldName'] == 'Roll':
                rolls.append(entry['value'])
            if entry['fieldName'] == 'Frame':
                frames.append(entry['value'])
            if entry['fieldName'] == 'Recording Technique':
                recordingTechniques.append(entry['value'])
            if entry['fieldName'] == 'Acquisition Date':
                acquisitionDates.append(entry['value'])
            if entry['fieldName'] == 'High Resolution Download Avail':
                hi_res_Available.append(entry['value'])
            if entry['fieldName'] == 'Image Type':
                imageTypes.append(entry['value'])
            if entry['fieldName'] == 'Quality':
                qualities.append(entry['value'])
            if entry['fieldName'] == 'Flying Height in Feet':
                altitudesFeet.append(float(entry['value']))
            if entry['fieldName'] == 'Photo ID':
                imageIds.append(entry['value'])
            if entry['fieldName'] == 'Focal Length':
                focalLengths.append(float(entry['value'].split(' ')[0]))
            if entry['fieldName'] == 'Center Latitude dec':
                centerLats.append(float(entry['value']))
            if entry['fieldName'] == 'Center Longitude dec':
                centerLons.append(float(entry['value']))
            if entry['fieldName'] == 'NW Corner Lat dec':
                NWlats.append(float(entry['value']))
            if entry['fieldName'] == 'NW Corner Long dec':
                NWlons.append(float(entry['value']))
            if entry['fieldName'] == 'NE Corner Lat dec':
                NElats.append(float(entry['value']))
            if entry['fieldName'] == 'NE Corner Long dec':
                NElons.append(float(entry['value']))
            if entry['fieldName'] == 'SE Corner Lat dec':
                SElats.append(float(entry['value']))
            if entry['fieldName'] == 'SE Corner Long dec':
                SElons.append(float(entry['value']))
            if entry['fieldName'] == 'SW Corner Lat dec':
                SWlats.append(float(entry['value']))
            if entry['fieldName'] == 'SW Corner Long dec':
                SWlons.append(float(entry['value']))
    scenceDict = {'entityId'          : entityIds, 
                  'agency'            : agencies,
                  'project'           : projects,
                  'roll'              : rolls,
                  'frame'             : agencies,
                  'recordingTechnique': recordingTechniques,
                  'acquisitionDate'   : acquisitionDates,
                  'hi_res_available'  : hi_res_Available,
                  'imageType'         : imageTypes,
                  'quality'           : qualities,
                  'altitudesFeet'     : altitudesFeet,
                  'imageId'           : imageIds,
                  'focalLength'       : focalLengths,
                  'centerLat'         : centerLats,
                  'centerLon'         : centerLons,
                  'NWlat'             : NWlats,
                  'NWlon'             : NWlons,
                  'NElat'             : NElats,
                  'NElon'             : NElons,
                  'SElat'             : SElats,
                  'SElon'             : SElons,
                  'SWlat'             : SWlats,
                  'SWlon'             : SWlons}
    df = pd.DataFrame(scenceDict)
    return df
    
def EE_login(username,
             password,
             m2mhost = 'https://m2m.cr.usgs.gov/api/api/json/stable/'):

    data = {'username' : username, 
            'password' : password}
    
    url = m2mhost + 'login'
    api_key = hipp.dataquery.EE_sendRequest(url, data)
    
    return api_key

def EE_sceneSearch(apiKey,
                   xmin,ymin,xmax,ymax,
                   startDate,endDate,
                   metadataType = 'full', #'summary', None
                   maxResults   = 2,
                   datasetName  = 'aerial_combin',
                   serviceUrl   = 'https://m2m.cr.usgs.gov/api/api/json/stable/'):
    """
    Example inputs:
    xmin       = -114.5
    ymin       = 48.2
    xmax       = -113.0
    ymax       = 49.2
    start_date = '1966-01-01'
    end_date   = '1966-12-10'     
    """
    print('Max records requested:',maxResults)
    print('\nBounds:\nxmin', xmin,'\nymin', ymin,'\nxmax',xmax,'\nymax',ymax)
    print('\nTime range:', startDate,'to', endDate)
    
    spatialFilter =  {'filterType' : "mbr",
                      'lowerLeft'  : {'latitude' : ymin, 'longitude' : xmax},
                      'upperRight' : {'latitude' : ymax, 'longitude' : xmin}}
    acquisitionFilter = {'start' : startDate, 'end' : endDate}
    datasetSearchParameters = {'datasetName' : datasetName,
                               'maxResults' : maxResults,
                               'sceneFilter' : {'spatialFilter'     : spatialFilter,
                                                'acquisitionFilter' : acquisitionFilter},
                               'metadataType': metadataType}
    scenes = hipp.dataquery.EE_sendRequest(serviceUrl + "scene-search", datasetSearchParameters, apiKey)
    print('\nRecords returned:', scenes['recordsReturned'])
    if scenes['recordsReturned'] == maxResults:
        print("\nmaxResults set to:", maxResults, 
              '\nIncrease this parameter to obtain additional records. API max 50,000.')
    
    return scenes['results']
    
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

def EE_stageForDownload(apiKey,
                        entityIds,
                        label        = 'test_download',
                        datasetName  = 'aerial_combin',
                        serviceUrl   = 'https://m2m.cr.usgs.gov/api/api/json/stable/'):
    
    # check what is available and get productIDs
    downloadOptionsParameters = {'datasetName' : datasetName,
                                 'entityIds' : entityIds}
    downloadOptions = hipp.dataquery.EE_sendRequest(serviceUrl + "download-options", 
                                                    downloadOptionsParameters, apiKey)
    entityIds_available = []
    downloads = []
    calibrationReports = []

    for product in downloadOptions:
        if product['available'] == True:
            if product['productName'] =='High Resolution Product':
                downloads.append({'entityId' : product['entityId'],
                                  'productId' : product['id']})
                entityIds_available.append(product['entityId'])
            if product['productName'] =='Camera Calibration File':
                calibrationReports.append({'entityId' : product['entityId'],
                                           'productId' : product['id']})
    print('Reqested images:', len(entityIds))
    print('Available images:',len(entityIds_available))
    if len(entityIds_available) == 0:
        print('No images available for downland.')
        sys.exit()
    diff = (list(list(set(entityIds)-set(entityIds_available)) + list(set(entityIds_available)-set(entityIds))))
    if len(diff) > 0:
        print('Unavailable images:')
        for i in [*diff]:
            print(i)
    
    # only download calibration report once.
    cal = list({v['productId']:v for v in calibrationReports}.values())
    if len(cal) > 0:
        print('Corresponding Calibration reports:', len(cal))
        downloads.extend(cal)
        
    print('Total files being requested:', len(entityIds)+len(cal))
    
    # request for staging
    downloadRequestParameters = {'downloads' : downloads,
                                 'label' : label}
    
    print('\nSending request to stage files for download in API location:', label)
    # TODO add spinner
    requestResults = hipp.dataquery.EE_sendRequest(serviceUrl + "download-request",
                                                   downloadRequestParameters, apiKey)
    
    fileNames = []
    urls      = []

    # handle previously requested files under different label
    if requestResults['duplicateProducts']:
        previouslyRequested_DownloadIds = list(requestResults['duplicateProducts'].keys())
        previouslyRequested_labels      = list(set(list(requestResults['duplicateProducts'].values())))
        if len(previouslyRequested_DownloadIds) > 0:
            print('\nRetrieving', 
                  len(previouslyRequested_DownloadIds), 
                  'previously requested files in API locations:',
                  *previouslyRequested_labels)
            for previousLabel in previouslyRequested_labels:
                downloadRetrieveParameters = {'label' : previousLabel}
                moreDownloadUrls = hipp.dataquery.EE_sendRequest(serviceUrl + "download-retrieve",
                                                                 downloadRetrieveParameters, apiKey)
                for downloadId in previouslyRequested_DownloadIds:
                    for i in moreDownloadUrls['available']:
                        if int(i['downloadId']) == int(downloadId):
                            if i['productName'] == 'USGS CAMERA CALIBRATION REPORT DOWNLOAD':
                                fileNames.append(i['entityId']+'_calibration_report.pdf')
                                urls.append(i['url'])
                            elif i['productName'] == 'AERIAL PHOTO SINGLE FRAME HIGH RESOLUTION DOWNLOAD':
                                fileNames.append(i['entityId']+'.tif.gz')
                                urls.append(i['url'])
            print('Retrieved', len(fileNames), 'previously requested files in API locations:', *previouslyRequested_labels)
            if len(fileNames) !=  len(previouslyRequested_DownloadIds):
                    print('Unable to find:',
                           len(previouslyRequested_DownloadIds) - len(fileNames),
                           'files. API says they should be in:',
                           *previouslyRequested_labels,
                           '¯\_(ツ)_/¯') 
                    #This issue is under investigation with the helpdesk... awaiting response.          

    # check for requests sent to new label
    downloadRetrieveParameters = {'label' : label}
    moreDownloadUrls = hipp.dataquery.EE_sendRequest(serviceUrl + "download-retrieve",
                                                     downloadRetrieveParameters, apiKey)
                                                     
    print('\n')
    if int(moreDownloadUrls['queueSize']) != 0:
        print('Staging',len(moreDownloadUrls['available']), 'new requests in API location:', label)
    while int(moreDownloadUrls['queueSize']) != 0:
        moreDownloadUrls = hipp.dataquery.EE_sendRequest(serviceUrl + "download-retrieve",
                                                         downloadRetrieveParameters, apiKey)
        # TODO if this takes to long should start thread to download available and pop from list
        if int(moreDownloadUrls['queueSize']) != 0:
            print('New requests in queue:', moreDownloadUrls['queueSize'])
            print('Retry in 30 seconds. Proceeding when queue = 0')
            time.sleep(30)


    for i in moreDownloadUrls['available']:
        if i['productName'] == 'Camera Calibration File':
            fileNames.append(i['entityId']+'_calibration_report.pdf')
            urls.append(i['url'])

        elif i['productName'] == 'High Resolution Product':
            fileNames.append(i['entityId']+'.tif.gz')
            urls.append(i['url'])
            
    print('\nAvailable and ready for download:', len(fileNames))
    
    if len(fileNames) != len(entityIds)+len(cal):
        print('\nWARNING: Missing files:',(len(entityIds)+len(cal)) - len(fileNames))
        for i in entityIds:
            if any(i in s for s in fileNames):
                pass
            else:
                print('Unable to find:', i)
                pass
#                 print('Unable to find:', i, 'in', *previouslyRequested_labels)

    return urls, fileNames


## ARCTICDATA.IO

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
    urls, file_names = df['urls'], df[file_name_column]
    
    hipp.dataquery.thread_downloads(output_directory, urls, file_names)
    
    hipp.utils.optimize_geotifs(output_directory)
    
    return output_directory

def NAGAP_pre_select_images(image_metadata,
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
    
    if not isinstance(image_metadata, type(pd.DataFrame())):
        df = pd.read_csv(image_metadata, dtype=object)
    else:
        df = image_metadata

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
        df = df[df['Month'] == str(month).zfill(2)]
        
    if not isinstance(day,type(None)):
        print('day:', day)
        df = df[df['Day'] == str(day).zfill(2)]
        
    df = df.reset_index(drop=True)
    
    if len(list(set(df['Roll'].values))) > 1:
        print('NOTE: Filter results contain multiple camera rolls:')
        for i in sorted(list(set(df['Roll'].values))):
            print(i)
        
    if len(list(set(df['Year'].values))) > 1:
        print('NOTE: Filter results contain multiple years:')
        for i in sorted(list(set(df['Year'].values))):
            print(i)
            
    if len(list(set(df['Year'].values))) == 1 and len(list(set(df['Month'].values))) > 1:
        print('NOTE: Filter results contain multiple months:')
        for i in sorted(list(set(df['Month'].values))):
            print(i)
        
    if not isinstance(output_directory,type(None)):
        out = os.path.join(output_directory,'targets.csv')
        df.to_csv(out, index=False)
        return out
    
    else:
        return df
        
