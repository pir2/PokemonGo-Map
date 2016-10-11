#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import requests
import time
import calendar
from datetime import datetime
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
    wh_pokemonids = args.webhook_ids

    # Create a search_worker_thread per account
    log.info('Starting search worker threads')

    # The real work starts here but will halt on pause_bit.set()
    # get the webhook area - borrowed from spawnpoint_only
    sp_dist = 0.07 * 2 * args.step_limit
    log.debug('Spawnpoint search radius: %f', sp_dist)
    # generate coords of the midpoints of each edge of the square
    south, west = get_new_coords(current_location, sp_dist, 180), get_new_coords(current_location, sp_dist, 270)
    north, east = get_new_coords(current_location, sp_dist, 0), get_new_coords(current_location, sp_dist, 90)

    swLat, swLng = south[0], west[1]
    neLat, neLng = north[0], east[1]
    
    while True:
        #get pokemon that are disappearing in the future
        #place pokemon into queue for webhook
        
        p = []
        
        if not wh_pokemonids:
            for p in Pokemon.get_active(swLat, swLng, neLat, neLng): 
                if p['encounter_id'] not in enc_ids_done:
                    wh_queue.put(('pokemon', {
                        'encounter_id': p['encounter_id'],
                        'spawnpoint_id': p['spawnpoint_id'],
                        'pokemon_id': p['pokemon_id'],
                        'latitude': p['latitude'],
                        'longitude': p['longitude'],
                        'disappear_time': calendar.timegm(p['disappear_time'].timetuple()),
                        'last_modified_time': '',
                        'time_until_hidden_ms': '',
                        'individual_attack': '',
                        'individual_defense': '',
                        'individual_stamina': '',
                        'move_1': '',
                        'move_2': ''
                    }))        
                    #add encounter id to enc_ids_done = {}
                    log.info('Webhook DB sent pokemon_id: {} to webhook'.format(p['pokemon_id']))
                    enc_ids_done.append(p['encounter_id'])
        else:
            for p in Pokemon.get_active_by_id(wh_pokemonids, swLat, swLng, neLat, neLng): 
                if p['encounter_id'] not in enc_ids_done:
                    wh_queue.put(('pokemon', {
                        'encounter_id': p['encounter_id'],
                        'spawnpoint_id': p['spawnpoint_id'],
                        'pokemon_id': p['pokemon_id'],
                        'latitude': p['latitude'],
                        'longitude': p['longitude'],
                        'disappear_time': calendar.timegm(p['disappear_time'].timetuple()),
                        'last_modified_time': '',
                        'time_until_hidden_ms': '',
                        'individual_attack': '',
                        'individual_defense': '',
                        'individual_stamina': '',
                        'move_1': '',
                        'move_2': ''
                    }))        
                    #add encounter id to enc_ids_done = {}
                    log.info('Webhook DB sent pokemon_id: {} to webhook'.format(p['pokemon_id']))
                    enc_ids_done.append(p['encounter_id'])
        
        #clean up old pokemon
        enc_ids_done = [done for done in enc_ids_done if done in p['encounter_id'].values()]
        
        #pause for 30s
        time.sleep(30)


