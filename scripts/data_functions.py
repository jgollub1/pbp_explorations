import numpy as np
import pandas as pd
import elo_538 as elo
from helper_functions import stats_52,tourney_52,normalize_name
import re
import datetime

def format_match_df(df,ret_strings=[],abd_strings=[]):
    df['pbp'] = [None]*len(df)
    df = df.rename(columns={'winner_name':'w_name','loser_name':'l_name'})
    grand_slam_d = dict(zip(['Australian Open','Roland Garros','Wimbledon','US Open'],[1]*4))
    df['is_gs'] = [name in grand_slam_d for name in df['tourney_name']]

    # Get dates into the same format
    df['tourney_date'] = [datetime.datetime.strptime(str(x), "%Y%m%d").date() for x in df['tourney_date']]
    df['match_year'] = [x.year for x in df['tourney_date']]
    df['match_month'] = [x.month for x in df['tourney_date']]
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", str(s)) for s in df['score']] # str(s) fixes any nans
    df['score'] = ['RET' if 'RET' in s else s for s in df['score']]
    df['w_swon'] = [df['w_1stWon'][i]+df['w_2ndWon'][i] for i in xrange(len(df))]
    df['l_swon'] = [df['l_1stWon'][i]+df['l_2ndWon'][i] for i in xrange(len(df))]
    df['w_rwon'] = df['l_svpt'] - df['l_swon']
    df['l_rwon'] = df['w_svpt'] - df['w_swon']
    df['w_rpt'] = df['l_svpt']
    df['l_rpt'] = df['w_svpt']

    # get rid of leading 0s in tourney_id
    df['tourney_id'] = ['-'.join(t.split('-')[:-1] + [t.split('-')[-1][1:]]) if t.split('-')[-1][0]=='0' else t \
                                    for t in df['tourney_id']]

    # get rid of matches involving a retirement
    df['score'] = ['ABN' if score.split(' ')[-1] in ('abandoned','ABN','ABD','DEF','def','unfinished','Walkover') \
                                else score for score in df['score']]
    ret_strings = ['ABN','DEF','In Progress','RET','W/O',' RET',' W/O','nan','walkover']
    ret_d = dict(zip(ret_strings,[1]*len(ret_strings)))
    df = df.loc[[i for i in range(len(df)) if df['score'][i] not in ret_d]]
    df = df.sort_values(by=['tourney_date','tourney_name','match_num'], ascending=True).reset_index()
    del df['index']; return df

def format_pbp_df(df,tour='atp'):
    df['w_name'] = np.where(df['winner'] == 0, df['server1'], df['server2'])
    df['l_name'] = np.where(df['winner'] == 0, df['server2'], df['server1'])
    df['w_name'] = [normalize_name(x,tour=tour) for x in df['w_name']]
    df['l_name'] = [normalize_name(x,tour=tour) for x in df['l_name']]
    df['date'] = pd.to_datetime(df['date'])
    df['match_year'] = [x.year for x in df['date']]
    df['match_month'] = [x.month for x in df['date']]
    df['date'] = [x.date() for x in df['date']]
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", s) for s in df['score']]
    return df

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
    if counts_i:
        df['w_elo_538'], df['l_elo_538'] = elo_1s, elo_2s
        df['w_s_elo_538'], df['l_s_elo_538'] = surface_elo_1s, surface_elo_2s
    else:
        df['w_elo'], df['l_elo'] = elo_1s, elo_2s
        df['w_s_elo'], df['l_s_elo'] = surface_elo_1s, surface_elo_2s
    return df


def generate_52_stats(df,start_ind):
    # track player match stats for every match since 2009 (we only need these for pbp matches)
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

def generate_tourney_stats(df,start_ind=0):
    tourney_stats = {}
    tourney_52_stats = np.zeros(len(df))
    for i, row in df.loc[start_ind:].iterrows():
        if row['tourney_name']=='Davis Cup' or row['w_swon']!=row['w_swon']:
            continue
        
        year,t_id = row['tourney_id'].split('-')
        year = int(year)
        match_stats = (row['w_swon']+row['l_swon'],row['w_svpt']+row['l_svpt'])
        if row['w_swon']!=row['w_swon']:
            tourney_52_stats[i] = 0 if tourney_stats[t_id].tourney_stats[1][1]==0 else \
                                tourney_stats[t_id].tourney_stats[1][0]/float(self.tourney_stats[1][1])
            
        elif t_id in tourney_stats:
            tourney_52_stats[i] = tourney_stats[t_id].update(year,match_stats)
        else:
            tourney_stats[t_id] = tourney_52(year)
            tourney_52_stats[i] = tourney_stats[t_id].update(year,match_stats)
        
    df['tourney_stats'] = tourney_52_stats
    return df

# have to correct the winner problem
def connect_df(match_df,pbp_df,col_d,start_year=2009):
    pbp_dict = {}; winner_dict = {}
    for i in xrange(len(pbp_df)):
        key = pbp_df['w_name'][i] +' ' +  pbp_df['l_name'][i] + ' ' \
            + str(pbp_df['match_year'][i]) + ' ' + pbp_df['score'][i]
        key = key+' '+str(pbp_df['match_month'][i]) if key in col_d else key
        if key in pbp_dict:
            continue
        pbp_dict[key] = pbp_df['pbp'][i]
        winner_dict[key] = pbp_df['winner'][i]

    # in case of a collision (about 10 cases), I only take the first match with that key
    c = 0
    pbps,winners = [],[]
    info = {}

    match_df = match_df[match_df['match_year']>=start_year]
    for i in match_df.index:
        key = match_df['w_name'][i] +' ' +  match_df['l_name'][i] + ' ' \
            +str(match_df['match_year'][i])+' '+match_df['score'][i]
        key = key+' '+str(match_df['match_month'][i]) if key in col_d else key
        if key in pbp_dict:
            c += 1
            pbps.append(pbp_dict[key])
            winners.append(winner_dict[key])
            if key in info:
                pbps[-1] = 'None'; winners[-1] = 'None'
                print 'collision'; print key + ' ' + str(match_df['match_month'][i])
            info[key] = 1
        else:
            pbps.append('None')
            #winners.append('NA')
            # we'll just make 'winner' a random 0 or 1 for now
            #winners.append(0)
            winners.append(np.random.choice([0,1]))
    print c
    match_df['pbp'] = pbps
    match_df['winner'] = winners

    #df = match_df[match_df['pbp']!='NA']
    #cols = df.columns.drop(['loser_id','winner_id'])
    df = match_df[match_df.columns.drop(['loser_id','winner_id'])]
    df = df.reset_index(); del df['index']

    # SET UP LOOP TO CHANGE W,L TO P0,P1
    for col in ['_name','_elo','_s_elo','_elo_538','_s_elo_538','_52_swon','_52_svpt','_52_rwon','_52_rpt']:
        df['p0'+col] = [df['l'+col][i] if df['winner'][i] else df['w'+col][i] for i in xrange(len(df))]
        df['p1'+col] = [df['w'+col][i] if df['winner'][i] else df['l'+col][i] for i in xrange(len(df))]

    df['elo_diff'] = [df['p0_elo'][i] - df['p1_elo'][i] for i in xrange(len(df))]
    df['s_elo_diff'] = [df['p0_s_elo'][i] - df['p1_s_elo'][i] for i in xrange(len(df))]
    df['tourney_name'] = [s if s==s else 'Davis Cup' for s in df['tourney_name']]
    return df

def generate_JS_stats(df):
    #### James-Stein estimators for 52-week serve and return percentages ####
    # calculate B_i coefficients for each player in terms of service points
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for label in ['p0','p1']:
        df[label+'_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])]
        df[label+'_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])]

    # fix this name later with find and replace...
    s_history = np.concatenate([df['p0_52_swon']/df['p0_52_svpt'],df['p1_52_swon']/df['p1_52_svpt']],axis=0)
    n = len(s_history)/2
    group_var = np.var(s_history)
    s_points = np.concatenate([df['p0_52_svpt'],df['p1_52_svpt']])
    sigma2_i = np.divide(p_hat*(1-p_hat),s_points,where=s_points>0)
    #sigma2_i = np.nan_to_num(p_hat*(1-p_hat)/np.concatenate([df['p0_52_svpt'],df['p1_52_svpt']]))
    tau2_hat = np.nanvar(s_history)
    B_i = sigma2_i/(tau2_hat+sigma2_i)
    df['B_i0_sv'],df['B_i1_sv'] = B_i[:n],B_i[n:]

    s_history[s_history!=s_history] = p_hat
    group_var = np.var(s_history)
    df['p0_s_pct_JS'] = df['p0_s_pct']+df['B_i0_sv']*(p_hat-df['p0_s_pct'])
    df['p1_s_pct_JS'] = df['p1_s_pct']+df['B_i1_sv']*(p_hat-df['p1_s_pct'])

    # repeat for return averages (slightly different tau^2 value)
    r_history = np.concatenate([df['p0_52_rwon']/df['p0_52_rpt'],df['p1_52_rwon']/df['p1_52_rpt']],axis=0)
    r_points = np.concatenate([df['p0_52_rpt'],df['p1_52_rpt']])
    sigma2_i = np.divide((1-p_hat)*p_hat,r_points,where=r_points>0)
    tau2_hat = np.nanvar(r_history)
    B_i = sigma2_i/(tau2_hat+sigma2_i)
    df['B_i0_r'],df['B_i1_r'] = B_i[:n],B_i[n:]

    r_history[r_history!=r_history] = 1-p_hat
    df['p0_r_pct_JS'] = r_history[:n]+df['B_i0_r']*((1-p_hat)-r_history[:n])
    df['p1_r_pct_JS'] = r_history[n:]+df['B_i1_r']*((1-p_hat)-r_history[n:])
    return p_hat,df