#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import requests
from .utils import get_args
from .models import Pokemon
from .search import get_new_coords
log = logging.getLogger(__name__)


def send_to_webhook(message_type, message):
    args = get_args()

    if not args.webhooks and not args.webhook_db_only:
        # what are you even doing here...
        return

    data = {
        'type': message_type,
        'message': message
    }

    for w in args.webhooks:
        try:
            requests.post(w, json=data, timeout=(None, 1))
        except requests.exceptions.ReadTimeout:
            log.debug('Response timeout on webhook endpoint %s', w)
        except requests.exceptions.RequestException as e:
            log.debug(e)


def wh_updater(args, q):
    # The forever loop
    while True:
        try:
            # Loop the queue
            while True:
                whtype, message = q.get()
                send_to_webhook(whtype, message)
                if q.qsize() > 50:
                    log.warning("Webhook queue is > 50 (@%d); try increasing --wh-threads", q.qsize())
                q.task_done()
        except Exception as e:
            log.exception('Exception in wh_updater: %s', e)

def webhook_overseer_thread(args, wh_queue, enc_ids_done, position):

    log.info('Webhook overseer starting')

    #parse_lock = Lock()
    threadStatus = {}

    current_location = [float(position[0]),float(position[1])]
    wh_pokemonids = [int(x) for x in args.webhook_ids]

    #threadStatus['Overseer'] = {}
    #threadStatus['Overseer']['message'] = "Initializing"
    #threadStatus['Overseer']['type'] = "Overseer"
    #threadStatus['Overseer']['method'] = "Webook"

    # Create a search_worker_thread per account
    log.info('Starting search worker threads')

    # A place to track the current location

    # The real work starts here but will halt on pause_bit.set()
    # get the webhook area - borrowed from spawnpoint_only
    sp_dist = 0.07 * 2 * args.step_limit
    log.debug('Spawnpoint search radius: %f', sp_dist)
    # generate coords of the midpoints of each edge of the square
    south, west = get_new_coords(current_location, sp_dist, 180), get_new_coords(current_location, sp_dist, 270)
    north, east = get_new_coords(current_location, sp_dist, 0), get_new_coords(current_location, sp_dist, 90)

    swLat, swLng = south, west
    neLat, neLng = north, east
    
    while True:
        #get pokemon that are disappearing in the future

        #place pokemon into queue for webhook
        
        for p in Pokemon.get_active_by_id(wh_pokemonids, swLat, swLng, neLat, neLng): 
            if p['encounter_id'] not in enc_ids_done['encounter_id']:
                wh_update_queue.put(('pokemon', {
                    'encounter_id': p['encounter_id'],
                    'spawnpoint_id': p['spawn_point_id'],
                    'pokemon_id': ['pokemon_id'],
                    'latitude': p['latitude'],
                    'longitude': p['longitude'],
                    'disappear_time': p['disappear_time'],
                    'last_modified_time': p['last_modified_timestamp_ms'],
                    'time_until_hidden_ms': p['time_till_hidden_ms']
                }))        
                #add encounter id to enc_ids_done = {}
                enc_ids_done.append(p)
        
        #clean up old pokemon
        new_enc_ids = [old for old in enc_ids_done if old['disappear_time'] > datetime.utcnow()]
        enc_ids_done = new_enc_ids
        
        #pause for 30s
        time.sleep(30)


