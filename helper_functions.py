# TO DO: 1) clean up, thoroughly, the existing code here
# 2) write enumerate_pbp, which will be handy if we use
# any future models that do rely on order
# get set_order is acting screwy

def enumerate_pbp(s,partition):
    sub_matches = ['']
    if partition == 'set':
        s = s.split('.')
        for i in range(1,len(s)):
            sub_matches.append('.'.join(s[:i])+'.')
        return sub_matches[:-1]
    
    elif partition=='game':
        s = s.split(';')
        s_new = []
        for i in range(len(s)):
            if '.' in s[i]:
                games = s[i].split('.')
                games[0] += '.'
                s_new += games
            else:
                s_new += [s[i]]
        for i in range(1,len(s_new)-1):
            sub_matches.append((';'.join(s_new[:i])+';').replace('.;','.'))
        return sub_matches[:-1]
    
    # now, divide into points
    elif partition=='point':
        s = s.split(';')
        s_new = []
        for i in range(len(s)):
            if '.' in s[i]:
                games = s[i].split('.')
                games[0] += '.'
                s_new += games
            else:
                s_new += [s[i]]
        for i in range(len(s_new)-1):
            up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')
            for k in range(len(s_new[i])):
                if s_new[i][k] not in ('.','/'):
                    sub_matches.append(up_til_now+s_new[i][:k+1]+';' if k==len(s_new[i])-1 else up_til_now+s_new[i][:k+1])
                elif s_new[i][k]=='.':
                    sub_matches[-1] += '.'
                #print 'sub: ', sub_matches
        return sub_matches[:-1]
            

# functions used to parse point-by-point tennis data
def simplify(s):
    s=s.replace('A','S');s=s.replace('D','R')
    sets = s.split('.')
    literal_s=''
    for k in range(len(sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server    
        games = sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        # now, iterate through every switch of serve
        for game in games:
            game = game.replace("S",str(server))
            game = game.replace("R",str((server+1)%2))
            literal_s += game
            server =(server+1)%2
    return literal_s

def find_pattern(s,pattern):
    # invert the pattern so as to count occurrences for second player
    inv_pattern = pattern.replace('0','x')
    inv_pattern = inv_pattern.replace('1','0')
    inv_pattern = inv_pattern.replace('x','1')
    pattern = list(pattern);inv_pattern = list(inv_pattern)
    
    times = [0,0]
    # convert string to list of chars so we can reassign characters after using them
    literal_s = list(s)
    # now, just find triples of '000' and '111' (for players 1 and 2 respectively)
    for i in range(len(literal_s) - len(pattern) + 1):
        if literal_s[i:i + len(pattern)]==pattern:
            times[0]+=1
            literal_s[i+len(pattern)-1] = 'x'
        elif literal_s[i:i + len(pattern)]==inv_pattern:
            times[1]+=1
            literal_s[i+len(pattern)-1] = 'x'  
    return times


def get_set_score(s):
    s=s.replace('A','S');s=s.replace('D','R')
    # split the string on '.' and count sets up to the second to last entry
    # (if the substring ends on a '.' the last element will be '')
    completed_sets = s.split('.')[:-1]
    p1_sets = 0; p2_sets = 0
    for k in range(len(completed_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server              
        games = completed_sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        final_server = (server + len(games) - 1)%2
        # award set to the player who won the last point of the set
        if final_server==0 and games[-1][-1]=='S' or final_server==1 and games[-1][-1]=='R':
            p1_sets += 1
        else:
            p2_sets += 1
    return [p1_sets,p2_sets]

def get_set_order(s):
    s=s.replace('A','S');s=s.replace('D','R')
    # split the string on '.' and count sets up to the second to last entry
    # (if the substring ends on a '.' the last element will be '')
    completed_sets = s.split('.')[:-1]
    sets = []
    for k in range(len(completed_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server              
        games = completed_sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        final_server = (server + len(games) - 1)%2
        # award set to the player who won the last point of the set
        if final_server==0 and games[-1][-1]=='S':
            sets += [0]
        elif final_server==1 and games[-1][-1]=='R':
            sets += [0]
        else:
            sets += [1]
    return sets


# gives the game score in current set of match substring 
# (will be [0,0] if match completed)
def get_game_score(s):
    s=s.replace('A','S');s=s.replace('D','R')
    add_d = {'0S':1,'1R':1,'1S':0,'0R':0}   
    # last entry in this will be '' if we split at the end of a set
    all_sets = s.split('.')
    p1_games, p2_games = 0,0
    for k in range(len(all_sets)):
        # iterate through the match so far just to see who's serving
        server = 0 if k==0 else next_server
        games = all_sets[k].split(';');length = len(games)
        # on the last (current) set, count the number of completed games
        if k==len(all_sets)-1:
            completed_games = all_sets[k].split(';')[:-1]
            #print completed_games
            for i in range(len(completed_games)):
                if i!=0:
                    server = (server+1)%2
                game = completed_games[i]
                if server==0 and game[-1]=='S':
                    p1_games += 1
                elif server==1 and game[-1]=='R':
                    #print 'hi'
                    p1_games += 1
                else:
                    p2_games += 1
        # update server with number of service switches; note if if there is tiebreak
        next_server = (server+1)%2 if length > 12 else (server + length)%2
    return [p1_games,p2_games]

# gets game order of entire match, with sets separated by periods
def get_game_order(s):
    s=s.replace('A','S');s=s.replace('D','R')  
    # last entry in this will be '' if we split at the end of a set
    all_sets = s.split('.')[:-1]
    game_s = ''
    for k in range(len(all_sets)):
        server = 0 if k==0 else next_server   
        #games = all_sets[k].split(';');length = len(games)
        game_s += get_game_order_sub(all_sets[k] + ';',server) + '.'
        next_server = (server+1)%2 if len(all_sets[k].split(';')) > 12 else (server + len(all_sets[k].split(';')))%2
    return game_s

# takes in s
def get_game_order_sub(s,server):
    games = s.split(';')[:-1]; game_s = ''
    for k in range(len(games)):
        if k==12:
            game_s += str(tbreak_winner(games[k],server))
        else:
            game_s += '0' if server==0 and games[k][-1]=='S' or server==1 and games[k][-1]=='R' else '1'
        server = 1 - server
    return game_s

def tbreak_winner(t_s,server):
    mini_games = t_s.split('/')
    for k in range(1,len(mini_games)):
        server = 1 - server
    return 0 if server==0 and mini_games[-1][-1]=='S' or server==1 and mini_games[-1][-1]=='R' else 1

def predictive_power(col,df):
    # find out how well col does at predicting match winners and losers
    times = 0
    even_indices = []
    for i in range(len(df)):
        if df[col][i][0] > df[col][i][1] and df['winner'][i]==0:
            times += 1
        elif df[col][i][0] < df[col][i][1] and df['winner'][i]==1:
            times += 1
        elif df[col][i][0] == df[col][i][1]:
            even_indices.append(i)
    return times/float(len(df)-len(even_indices)), len(df)-len(even_indices)





#S = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSSSS.SRRSSRSS;SSSS;RSRSRR;RSRSSS;SSSRS;SSRSS;SSSS;SSSRS;SSSRRRRSR.'
S = 'SSSS;RRRR;RSSSS;RSSRRSRSRR;SRRSSS;RRSSRR.SSRRSS;RSSSS;SRRRR;RSRRSR;SRSSRS;SSSRS;SSSS;RRRSSR;RRSSSRSRS.'
S1 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R;'
S2 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;'
S3 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;S'
S4 = 'SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSS'
#print S
#print get_game_score(S)
#print get_game_score(S2)
#print get_game_score(S3)
#print get_game_score(S4)
#print get_game_order(S)
#print get_game_order_sub(S1,0)
#print get_set_order(S2)


#x,y = 3,4
#x+=1 if 3>2 else y
#print x,y

#S_full = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSSSS.SRRSSRSS;SSSS;RSRSRR;RSRSSS;SSSRS;SSRSS;SSSS;SSSRS;SSSRRRRSR.'
#for a in enumerate_pbp(S_full,'point'):
#    print a
#print ';'.join(['SSSS'])
