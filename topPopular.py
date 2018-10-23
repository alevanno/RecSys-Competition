#! python

import os
import os.path
import sys
import csv
import logging
import operator
import json

log = logging.getLogger(__name__)

log.setLevel(logging.INFO)
sh = logging.StreamHandler()
log.addHandler(sh)

datadir = ('..', 'Competition')

def foobar():
    with open(os.path.join(*datadir, 'tracks.csv')) as f:
        data = csv.reader(f)
        next(data)  # skip header
        track_list, album_list, artist_list, _ = zip(*data)

    log.debug(album_list[1])
    log.debug(track_list[1])
    log.debug(artist_list[1])


    albumlist_uniq = set(album_list)
    tracklist_uniq = set(track_list)
    artistlist_uniq = set(artist_list)

    num_album = len(albumlist_uniq)
    num_track = len(tracklist_uniq)
    num_artist = len(artistlist_uniq)

    log.info("Num of artists: %s", num_artist)
    log.info("Num of album: %s", num_album)
    log.info("Num of tracks: %s", num_track)


def tracks():
    """
    Returns a tuple containig this two dictionaries
    tracks_counter = {track_id: num_plays(presences in playlists)}
    playlists = {playlist_id: [trackid, trackid, trackid]}
    """
    with open(os.path.join(*datadir, 'train.csv')) as f:
        data = csv.reader(f)
        next(data)  # skip header
        data = list(data)
    
    tracks_counter = {}
    playlists = {}
    for pl, tr in data:
        pl = int(pl)
        tr = int(tr)
        try:
            tracks_counter[tr] += 1
        except KeyError:
            tracks_counter[tr] = 1
        try:
            playlists[pl].append(tr)
        except KeyError:
            playlists[pl] = [tr]
    log.debug(tracks_counter)
    log.debug(playlists)        

    return tracks_counter, playlists


def top_tracks(tracks: dict, num=10) -> list:
    """
    Returns a list containing the "num" top popular tracks 
    [(track_id, counter), (track_id, counter), ...]
    """
    m = sorted(tracks.items(), key=operator.itemgetter(1), reverse=True)[:num]
    log.info("top tracks: %s", m)
    return m


def target_playlist() -> list:
    """
    Returns a list of int containing the target playlist ids
    """
    with open(os.path.join(*datadir, 'target_playlists.csv')) as f:
        data = csv.reader(f)
        next(data)  # skip header
        data = list(zip(*data))[0]
        data = [int(x) for x in data]
    return data


def top_popular(tracks_counter, playlists: dict) -> dict:
    """
    {playlist_id: [trackid, trackid]}
    """
    max_length_playlist = len(max(playlists.values(), key=len))
    log.info("max_length_playlist: %s", max_length_playlist+10)
    tt = top_tracks(tracks_counter, max_length_playlist)
    tplaylists = target_playlist()
    recommends = {}
    for p in tplaylists:
        recommends[p] = []
        for t, _ in tt:
            try:
                if t not in playlists[p]:
                    recommends[p].append(t)
            except KeyError:
                # Playlist %s not in training data
                recommends[p].append(t)
            if len(recommends[p]) >= 10:
                break
    return recommends



if __name__ == '__main__':
    tracks_counter, playlists = tracks()
    recommends = top_popular(tracks_counter, playlists)
    
    with open('topPop.csv', 'w') as f:
        f.write('playlist_id,track_ids\n')
        for i in sorted(recommends):
            f.write('{},{}\n'.format(i, ' '.join([str(x) for x in recommends[i]])))
