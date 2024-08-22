import numpy as np
import pandas as pd
import copy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import Stream, Inventory
from obspy.clients.fdsn.client import Client
from obspy.clients.fdsn import RoutingClient
from obspy.core.event import Catalog
from obspy.clients.fdsn.header import URL_MAPPINGS

def locate_z_index(stream):
    """
    Finds the index of the Z component in a seismic stream.
    
    :param stream: ObsPy Stream object
    :return: Index of the Z component trace
    """
    z_component_index = None
    for i, trace in enumerate(stream):
        if trace.stats.channel.endswith("Z"):
            return i
        
def get_stream(agency, starttime, endtime, station, channels, attach_response=True, network='??', location='*',resample=False):
    """
    Retrieves waveform data from a specified agency for given parameters.
    
    :param agency: Data center or agency code
    :param starttime: Start time for data request
    :param endtime: End time for data request
    :param station: Station code
    :param channels: Channel codes
    :param attach_response: Whether to attach instrument response
    :param network: Network code
    :param location: Location code
    :param resample: Resampling frequency (if specified)
    :return: ObsPy Stream object containing requested waveforms
    """
    st = Stream()
    if agency== "iris-federator" or agency == "eida-routing":
        client = RoutingClient(agency)
    else:
        client = Client(agency)

    temp = client.get_waveforms(network=network,
               station=station,
               channel=channels,
               starttime=starttime,
               endtime=endtime,
               location=location,
               attach_response=attach_response)

    st.append(temp)
    if resample:
        for i,tr in enumerate(st):
            st[i] = tr.resample(resample)
            
    return st

# def get_stations(fdsn_urls,starttime=None, endtime=None, channels=None, networks=None, 
#                  minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, 
#                  latitude=None,longitude=None, minradius=None, maxradius=None, 
#                  level = 'channel',matchtimeseries=True, bulk=None, **kwargs):
#     """
#     Retrieves station metadata from multiple FDSN web services.
    
#     :param fdsn_urls: List of FDSN service URLs
#     :param starttime: Start time for station search
#     :param endtime: End time for station search
#     :param channels: Channel codes
#     :param networks: Network codes
#     :param minlatitude: Minimum latitude for bounding box
#     :param maxlatitude: Maximum latitude for bounding box
#     :param minlongitude: Minimum longitude for bounding box
#     :param maxlongitude: Maximum longitude for bounding box
#     :param latitude: Central latitude for circular search
#     :param longitude: Central longitude for circular search
#     :param minradius: Minimum radius for circular search
#     :param maxradius: Maximum radius for circular search
#     :param level: Level of detail for station metadata
#     :param matchtimeseries: Whether to match time series
#     :param bulk: Bulk request parameters
#     :param kwargs: Additional keyword arguments
#     :return: ObsPy Inventory object containing station metadata
#     """    
#     inventory = Inventory()
#     for url in fdsn_urls:
#         try:
#             if url in ["iris-federator","eida-routing"]:
#                 client = RoutingClient(url)
#                 temp = client.get_stations_bulk(bulk=bulk,**kwargs)
#             else:    
#                 client = Client(base_url=url)
#                 temp = client.get_stations(bulk=bulk,**kwargs)
        
        
#             inventory.extend(temp)
#         except:
#             continue
#     return inventory



def inventory_to_dataframe(inventory, agency='__', filter_network=[], level='channel'):
    """
    Processes an ObsPy Inventory object and converts it to a pandas DataFrame.
    
    :param inventory: ObsPy Inventory object
    :param agency: Agency code
    :param filter_network: List of network codes to filter
    :param level: Level of detail ('station' or 'channel')
    :return: pandas DataFrame containing processed inventory data
    """
    rows = []
    for network_item in inventory:
        net = network_item.code
        if net not in filter_network:
            for station_item in network_item:
                
                if level == 'station':
                    row = {'agency': str(agency),
                           'sta': str(station_item.code),
                           'net': str(net),
                           'lat': float(station_item.latitude),
                           'lon': float(station_item.longitude),
                           'elevation': float(station_item.elevation),
                           'starttime': str(station_item.start_date),
                           'endtime': str(station_item.end_date),
                           'sitename': str(station_item.site.name),
                           # 'access_status': str(station_item.restricted_status)
                          }
                    rows.append(row)
                if level =='channel':
                    for channel_item in station_item: #[0][0][0]
                        row ={  'agency': str(agency),
                                'net': str(net),
                                'sta': str(station_item.code),
                                'chan': str(channel_item.code),
                                'loc' : str(channel_item.location_code),
                                'lat' : float(channel_item.latitude),
                                'lon' : float(channel_item.longitude),
                                'elevation' : float(channel_item.elevation),
                                'starttime' : str(channel_item.start_date),
                                'endtime' : str(channel_item.end_date),
                                # 'sensor' : str(channel_item.sensor.description) if channel_item.sensor is not None else str('NaN'),
                                'sitename':str(station_item.site.name),
                                # 'access_status': str(station_item.restricted_status)
                             }

                        rows.append(row)
    # temp_df = pd.json_normalize(temp)            
    df = pd.DataFrame(rows)
    return df



def get_random_stream(time_interval=360, network='IU', station='ANMO', channel='BHZ', 
					location='00', agency='IRIS',output_stream=True,attach_response=False,
					random_state=1000, minyear=2018, maxyear=2024):
    """
    Downloads random stream data from a specified agency within given parameters.
    
    :param time_interval: Duration of stream data in seconds
    :param network: Network code
    :param station: Station code
    :param channel: Channel code
    :param location: Location code
    :param agency: Data center or agency code
    :param output_stream: Whether to output as an ObsPy Stream
    :param attach_response: Whether to attach instrument response
    :param random_state: Random state for reproducibility
    :param minyear: Minimum year for random time selection
    :param maxyear: Maximum year for random time selection
    :return: ObsPy Stream object containing noise data
    """
    rand_gen = np.random.RandomState(random_state)    
    download_data = True 
    count = 0
    while download_data:
        #generate random datetime(UTC)
        day = int(np.floor(rand_gen.uniform(low=1, high=29, size=None)))
        month = int(np.floor(rand_gen.uniform(low=1, high=12, size=None)))
        year = int(np.floor(rand_gen.uniform(low=minyear, high=maxyear, size=None)))
        hour = int(np.floor(rand_gen.uniform(low=0, high=24, size=None)))
        minute = int(np.floor(rand_gen.uniform(low=0, high=60, size=None)))
        second = int(np.floor(rand_gen.uniform(low=0, high=60, size=None)))
        starttime = UTCDateTime(year,month,day,hour,minute,second)
        endtime = starttime + time_interval
        
        st_noise = []
        try:
            st_noise = get_stream(agency, starttime, endtime, station, channel,
                                  attach_response=attach_response, network=network, 
                                  location=location,resample=False)
        except:
            count += 1

        if len(st_noise) > 0:
            download_data = False
            return st_noise
        else:
            if count > 50:
                raise ValueError('Failed to download random stream')
            else:
                continue 
                
def get_obspy_catalog(fdsn_urls=['USGS'], starttime=None, endtime=None, 
                      minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, 
                      latitude=None, longitude=None, minradius=None, maxradius=None, mindepth=None, 
                      maxdepth=None, minmagnitude=None, maxmagnitude=None, **kwargs):
    """
    Retrieves earthquake catalog data from multiple FDSN web services.
    
    :param fdsn_urls: List of FDSN service URLs
    :param starttime: Start time for event search
    :param endtime: End time for event search
    :param minlatitude: Minimum latitude for bounding box
    :param maxlatitude: Maximum latitude for bounding box
    :param minlongitude: Minimum longitude for bounding box
    :param maxlongitude: Maximum longitude for bounding box
    :param latitude: Central latitude for circular search
    :param longitude: Central longitude for circular search
    :param minradius: Minimum radius for circular search
    :param maxradius: Maximum radius for circular search
    :param mindepth: Minimum depth for event search
    :param maxdepth: Maximum depth for event search
    :param minmagnitude: Minimum magnitude for event search
    :param maxmagnitude: Maximum magnitude for event search
    :param kwargs: Additional keyword arguments
    :return: ObsPy Catalog object containing event data
    """
    cat = Catalog()
    for url in fdsn_urls:
        if url == 'iris-federator' or url == 'eida-routing':
            client = RoutingClient(url)
        else:
            client = Client(url)

        temp = client.get_events(starttime=starttime, endtime=endtime, 
                          minlatitude=minlatitude, maxlatitude=maxlatitude, 
                          minlongitude=minlongitude, maxlongitude=maxlongitude, 
                          latitude=latitude, longitude=longitude, 
                          minradius=minradius, maxradius=maxradius, mindepth=mindepth, 
                          maxdepth=maxdepth, minmagnitude=minmagnitude, maxmagnitude=maxmagnitude,
                          **kwargs)

        cat.extend(temp)
    return cat

def obspy_catalog_to_dataframe(catalog):
    """
    Converts an ObsPy Catalog object to a pandas DataFrame.
    
    :param catalog: ObsPy Catalog object
    :return: pandas DataFrame containing event data
    """
    df = pd.DataFrame([])
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        latitude = origin.latitude
        longitude = origin.longitude
        origintime = origin.time
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        magnitude = np.round(magnitude.mag, 1)
        depth_meters = origin.depth

        temp = { 'lat':latitude,
                'lon':longitude,
                'origintime':origintime,
                'magnitude':magnitude,
                'depth' : depth_meters/1000.
               }
        tempdf = pd.json_normalize(temp)
        df = pd.concat([df, tempdf])
    return df

def generate_bulk(stations, eventloc):
    """
    Generates a bulk request for waveform data based on station list and event location.
    
    :param stations: List of station codes
    :param eventloc: Event location object
    :return: List of tuples for bulk data request
    """
    bulk = []
    for station in stations:
        bulk.append(("??", station, "*", "*", UTCDateTime(eventloc.time), UTCDateTime(eventloc.time) + 3600))
    return bulk

def chan_priority(waveform_df,uniqsta_df,j):
    """
    Station channel prioritization.

    :param waveform_df: DataFrame with waveform information.
    :param uniqsta_df: DataFrame with unique station information.
    :param j: Index of the unique station in uniqsta_df.
    :return: Channel to use for the station.
    """
    BH = (waveform_df.index[(waveform_df['sta'] == str(uniqsta_df.loc[j]['uniqsta'])) & (waveform_df['chan'].str.contains('BH'))].tolist())
    HH = (waveform_df.index[(waveform_df['sta'] == str(uniqsta_df.loc[j]['uniqsta'])) & (waveform_df['chan'].str.contains('HH'))].tolist())
    EH = (waveform_df.index[(waveform_df['sta'] == str(uniqsta_df.loc[j]['uniqsta'])) & (waveform_df['chan'].str.contains('EH'))].tolist())
    EN = (waveform_df.index[(waveform_df['sta'] == str(uniqsta_df.loc[j]['uniqsta'])) & (waveform_df['chan'].str.contains('EN'))].tolist())

    if len(BH) >= len(HH) and len(BH) >= len(EH) and len(BH) >= len(EN):
        usechan = 'BH'
    elif len(BH) < len(HH) and len(HH) >= len(EH) and len(HH) >= len(EN):
        usechan = 'HH'
    elif len(HH) < len(EH) and len(EH) > len(BH) and len(EH) >= len(EN):
        usechan = 'EH'
    elif len(HH) < len(EN) and len(EN) > len(BH) and len(EN) >= len(EH):
        usechan = 'EN'


    return usechan

def validate_horizontals_for_rotation(instream):
    """
    Validate the Horizontal Channels in a stream for rotation.
    Same start time, same number of points.

    :param instream: Input stream.
    :return: Tuple with a boolean indicating if the channels are valid for rotation, and the indices of the N and E components.
    """
    s = copy.deepcopy(instream)
    ntr = len(s.traces)
    ok = False
    h = [] # the indices of horizontal traces
    ntr = len(instream.traces)
    if(ntr >= 2):
        # identify the horizontals
        i = 0;
        components = [tr.stats.channel[-1] for tr in instream]
        components = np.array(components)
        idxE = np.where(components=='E')
        idxE = np.asscalar(idxE[0])
        idxN = np.where(components=='N')
        idxN = np.asscalar(idxN[0])
    
        if(ntr >= 2):
            h1 = instream.traces[idxN]
            h2 = instream.traces[idxE]
            dt = np.fabs(h1.stats.starttime - h2.stats.starttime)
            if(dt <= 0.1 * h1.stats.delta):
                if(h1.stats.npts == h2.stats.npts):
                    if(h1.stats.sampling_rate == h2.stats.sampling_rate):
                        ok = True
                    else:
                        print( 'Sampling rates not equal.')
                else:
                    print('npts not equal.')
            else:
                print('Start times not equal.')

    return ok, idxN, idxE

def get_trace_index(instream, runOpt):
    """
    Get the indices of the traces based on the run option.

    :param instream: Input stream.
    :param runOpt: Run option ('ENZ', 'RTZ', or 'Z').
    :return: Tuple with the indices of the traces based on the run option.
    """
    components = [tr.stats.channel[-1] for tr in instream]
    components = np.array(components)
    if runOpt == 'ENZ':
        idxE = np.where(components=='E')
        idxE = np.asscalar(idxE[0])
        idxN = np.where(components=='N')
        idxN = np.asscalar(idxN[0])
        idxZ = np.where(components=='Z')
        idxZ = np.asscalar(idxN[0])
        return idxE, idxN, idxZ
    if runOpt == 'RTZ':
        idxR = np.where(components=='R')
        idxR = np.asscalar(idxR[0])
        idxT = np.where(components=='T')
        idxT = np.asscalar(idxT[0])
        idxZ = np.where(components=='Z')
        idxZ = np.asscalar(idxZ[0])
        return idxR, idxT, idxZ
    if runOpt == 'Z':
        idxZ = np.where(components=='Z')
        idxZ = np.asscalar(idxZ[0])
        return idxZ
    
def rotate_horizontals_to_rt(instream, baz_in, ok, idxN, idxE):

    """
    Rotate the Horizontal Channels in a stream.

    :param instream: Input stream.
    :param baz_in: Back azimuth in degrees.
    :param ok: Boolean indicating if the channels are valid for rotation.
    :param idxN: Index of the N component.
    :param idxE: Index of the E component.
    :return: Stream with rotated traces.
    """
    #instream = st
    s = copy.deepcopy(instream)
    #
    ntr = len(s.traces)
    #
    #ok, idxN, idxE = validate_horizontals_for_rotation(instream)
    
    #
    if(ok == False):
        return None
    #
    d2r = np.arccos(-1.0)/180.0;
    baz = baz_in * d2r #FIX
    cmpaz1 = 0 * d2r #N
    cmpaz2 = 90 * d2r #E
    
    cb = np.cos(baz)
    sb = np.sin(baz)
    ca1 = np.cos(cmpaz1)
    sa1 = np.sin(cmpaz1)
    ca2 = np.cos(cmpaz2)
    sa2 = np.sin(cmpaz2)
    
    c1 = np.copy(s.traces[idxN].data)
    c2 = np.copy(s.traces[idxE].data)
    # account for component orientations
    east  = sa1*c1 + sa2*c2
    north = ca1*c1 + ca2*c2;
    # radial, then transverse
    s.traces[idxN].data = -sb * east - cb * north
    s.traces[idxE].data = -cb * east + sb * north
    
    baz = baz_in #FIXED
    r_az = baz - 180
    if(r_az > 360):
        r_az -= 360 
    # this will work as long as channel codes only differ after
    # the first two characters
    prefix = s.traces[-1].stats.channel[0:2]
    s.traces[idxN].stats.channel = prefix+'R'
    st.traces[idxN].stats.sac.kcmpnm = prefix+'R'
    #
    t_az = r_az + 90
    if(t_az > 360):
        t_az -= 360
    #
    s.traces[idxE].stats.channel = prefix+'T'
    st.traces[idxE].stats.sac.kcmpnm = prefix+'T'
    
    return s
