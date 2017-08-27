import numpy as np
import pandas as pd
import elo_538 as elo
from helper_functions import stats_52


# takes in a dataframe of matches in atp/wta format and returns the dataframe with elo columns
def generate_elo(df,counts_i):
    players_list = np.union1d(df.w_name, df.l_name)
    players_elo = dict(zip(list(players_list), [elo.Rating() for __ in range(len(players_list))]))
    surface_elo = {}
    for surface in ('Hard','Clay','Grass'):
        surface_elo[surface] = dict(zip(list(players_list), [elo.Rating() for __ in range(len(players_list))])) 

    elo_1s, elo_2s = [],[]
    surface_elo_1s, surface_elo_2s = [],[]
    elo_obj = elo.Elo_Rater()

    # update player elo from every recorded match
    for i, row in df.iterrows():
        surface = row['surface']; is_gs = row['is_gs']
        # append elos, rate, update
        w_elo,l_elo = players_elo[row['w_name']],players_elo[row['l_name']]
        elo_1s.append(w_elo.value);elo_2s.append(l_elo.value)    
        elo_obj.rate_1vs1(w_elo,l_elo,is_gs,counts=counts_i)


        surface_elo_1s.append(surface_elo[surface][row['w_name']].value if surface in ('Hard','Clay','Grass') else w_elo.value)
        surface_elo_2s.append(surface_elo[surface][row['l_name']].value if surface in ('Hard','Clay','Grass') else l_elo.value)
        if surface in ('Hard','Clay','Grass'):
            new_elo1, new_elo2 = elo_obj.rate_1vs1(surface_elo[surface][row['w_name']],surface_elo[surface][row['l_name']],is_gs,counts=counts_i)

    # add columns
    df['w_elo'], df['l_elo'] = elo_1s, elo_2s
    df['w_s_elo'], df['l_s_elo'] = surface_elo_1s, surface_elo_2s
    return df


def generate_52_stats(df):
    # track player match stats for every match since 2009 (we only need these for pbp matches)
    start_ind = 136146
    players_stats = {}
    # an array containing 2x1 arrays for winner and loser's previous 12-month serve performance over all matches in df
    match_52_stats = np.zeros([2,len(df),4])
    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():    
        date = row['match_year'],row['match_month']           
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = stats_52(date)
            # store serving stats prior to match
            match_52_stats[k][i] = np.sum(players_stats[row[label+'_name']].last_year,axis=0)
            # update serving stats if no
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:    
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                players_stats[row[label+'_name']].update(date,match_stats)
            match_52_stats[k][i] = np.sum(players_stats[row[label+'_name']].last_year,axis=0)

    for k,label in enumerate(w_l):
        df[label+'_52_swon'] = match_52_stats[k][:,0]
        df[label+'_52_svpt'] = match_52_stats[k][:,1]
        df[label+'_52_rwon'] = match_52_stats[k][:,2]
        df[label+'_52_rpt'] = match_52_stats[k][:,3]
    return df


