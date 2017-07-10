# the idea here is to make one or a couple functions which get set,
# game, point score, service breaks, p_serve/p_return so far
# however, don't spend too much time trying to optimize this when you
# could be exploring more features/models

import numpy as np
# right now it works EXCEPT FOR TIEBREAKS
# use a converter from S/RS/SS form to one game of 'S'
# eventually, this will enumerate games, sets, points, server_points all 
# in this function
# TO DO: format this so the 'tiebreak' condition does not activate in 
# grand slam, final set conditions
# now, just specify final_set_extend==1 for the first three grand slams/Davis Cup

def enumerate_pbp_2(s,columns,final_set_extend=0):

    # find the number of S,R,D,A characters and use this to initialize
    # all columns as npy arrays of this length
    length = len(s.replace('.','').replace('/','').replace(';',''))

    sub_matches = ['']; sub_sets = [0]
    t_points,p0_tp, p1_tp = [0,0],[0],[0]
    s_points,p0_sp, p1_sp = [0,0],[0],[0]
    sw_points,p0_swp, p1_swp = [0,0],[0],[0]
    points,p0_p, p1_p = [0,0],[0],[0]
    games,p0_g, p1_g = [0,0],[0],[0]
    sets,p0_s, p1_s = [0,0],[0],[0]
    server = 0; servers=[]
    
    # divides into s_new, array of games
    s = s.split(';'); s_new = []
    for i in range(len(s)):
        if '.' in s[i]:
            game_str = s[i].split('.')
            game_str[0] += '.'
            s_new += game_str
        else:
            s_new += [s[i]]

    # iterate through each game, a point at a time
    for i in range(len(s_new)-1):
        # update server; up_til_now is everything that has elapsed
        server = (server+1)%2 if i>0 else server
        up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')

        # update points and sets if there is a tiebreaker
        if games[0]==games[1]==6 and not (p0_s==p1_s==2 and final_set_extend):
            t_server = server
            mini_games = s_new[i].split('/')
            for j in range(len(mini_games)):
                for l in range(len(mini_games[j])):
                    if mini_games[j][l]!='.':
                        winner = int(mini_games[j][l] in ('S','A') and t_server==1 or mini_games[j][l] in ('R','D') and t_server==0) 
                        points[winner]+=1
                        t_points[winner]+=1
                        if winner==server: sw_points[winner]+=1
                        s_points[server]+=1
                        sub_m = up_til_now+'/'.join(mini_games[:j])+'/'+mini_games[j][:l+1] if j>0 else up_til_now+mini_games[j][:l+1]
                        sub_matches.append(sub_m)
                        p0_p.append(points[0]);p1_p.append(points[1])
                        p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                        p0_g.append(games[0]);p1_g.append(games[1])
                        p0_s.append(sets[0]);p1_s.append(sets[1])
                        p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                        p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                        servers.append(t_server)
                    else:
                        sets[winner]+=1
                        points, games = [0,0], [0,0]
                        p0_s[-1],p1_s[-1]=sets[0],sets[1]
                        p0_g[-1],p1_g[-1]=games[0],games[1]
                        p0_p[-1],p1_p[-1]=points[0],points[1]
                        sub_matches[-1] += '.'
                t_server = 1 - t_server

        # otherwise 
        else:
            for k in range(len(s_new[i])):
                if s_new[i][k]=='/':
                    print 'ERROR; this should be TIEBREAK'
                    return 0

                if s_new[i][k]!='.':    
                    winner = int(s_new[i][k] in ('S','A') and server==1 or s_new[i][k] in ('R','D') and server==0) 
                    points[winner]+=1
                    t_points[winner]+=1
                    if winner==server: sw_points[winner]+=1
                    s_points[server]+=1
                    if k==len(s_new[i])-1:
                        sub_matches.append(up_til_now+s_new[i][:k+1]+';')
                        games[winner]+=1
                        points = [0,0]
                    else:
                        sub_matches.append(up_til_now+s_new[i][:k+1])

                    p0_p.append(points[0]);p1_p.append(points[1])
                    p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                    p0_g.append(games[0]);p1_g.append(games[1])
                    p0_s.append(sets[0]);p1_s.append(sets[1])
                    p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                    p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                    servers.append(server)

                # backtrack and update previous entries if we finished a set; reset points and games
                elif s_new[i][k]=='.':
                    sets[winner]+=1
                    points, games = [0,0], [0,0]
                    p0_s[-1],p1_s[-1]=sets[0],sets[1]
                    p0_g[-1],p1_g[-1]=games[0],games[1]
                    p0_p[-1],p1_p[-1]=points[0],points[1]
                    sub_matches[-1] += '.'
        
        points = [0,0]
        columns = [[item]*len(servers) for item in columns]
        # have to add elo_diffs
    return columns+[sub_matches[:-1],servers,p0_s[:-1],p1_s[:-1],p0_g[:-1],p1_g[:-1],p0_p[:-1],p1_p[:-1],p0_tp[:-1],p1_tp[:-1],p0_swp[:-1],p0_sp[:-1],p1_swp[:-1],p1_sp[:-1]]


#mini_games = t_s.split('/')
#for k in range(1,len(mini_games)):
#    server = 1 - server








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
            for i in range(len(completed_games)):
                if i!=0:
                    server = (server+1)%2
                game = completed_games[i]
                if server==0 and game[-1]=='S':
                    p1_games += 1
                elif server==1 and game[-1]=='R':
                    p1_games += 1
                else:
                    p2_games += 1
        # update server with number of service switches; note if if there is tiebreak
        next_server = (server+1)%2 if length > 12 else (server + length)%2
    return [p1_games,p2_games]

def get_point_score(s):
    s=s.replace('A','S');s=s.replace('D','R')
    all_sets = s.split('.')
    points = [0,0]
    for k in range(len(all_sets)):
        # iterate through the match so far just to see who's serving
        server = 0 if k==0 else next_server
        games = all_sets[k].split(';');length = len(games)
        if k==len(all_sets)-1:
            completed_games = all_sets[k].split(';')[:-1]
            current_server = (server+len(completed_games))%2
            last_game = all_sets[k].split(';')[-1]
            points[current_server] = last_game.count('S')
            points[1-current_server] = last_game.count('R')
            return points
        # update server with number of service switches; note if if there is tiebreak
        next_server = (server+1)%2 if length > 12 else (server + length)%2
    return [p1_points,p2_points]

def service_breaks(s):
    ## return the service break advantage ##
    s=s.replace('A','S');s=s.replace('D','R')
    all_sets = s.split('.'); p1_games, p2_games = 0,0
    for k in range(len(all_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server 
        games = all_sets[k].split(';');length = len(games)
        next_server = (server+1)%2 if length>12 else (server + len(games))%2   
        if k==len(all_sets)-1:
            completed_games = all_sets[k].split(';')[:-1]
            for i in range(len(completed_games)):
                if i!=0: 
                    server = (server+1)%2 
                game = completed_games[i]
                if server==0 and game[-1]=='S' or server==1 and game[-1]=='R':
                    p1_games += 1
                else:
                    p2_games += 1
        next_server = (server+1)%2 if length > 12 else (server + length)%2
    server = (server+1)%2
    if server==0:
        break_adv = math.ceil(float(p1_games-p2_games)/2)
    else:
        break_adv = math.ceil(float(p1_games-p2_games-1)/2)
    return int(break_adv)

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

def tbreak_winner(t_s,server):
    mini_games = t_s.split('/')
    for k in range(1,len(mini_games)):
        server = 1 - server
    return 0 if server==0 and mini_games[-1][-1]=='S' or server==1 and mini_games[-1][-1]=='R' else 1

# include a pre-match prediction
def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask, reuse_split=None, score_func=None, n_folds=5, n_jobs=1):
    subdf=indf[featurenames]
    print 'type: ',str(type(clf)).split('.')[-1].split("'")[0]

    Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
    #print len(Xtrain),len(ytrain)
    clf=clf.fit(Xtrain, ytrain)
    probs_train,probs_test = clf.predict_proba(Xtrain),clf.predict_proba(Xtest)
    train_loss, test_loss = log_loss(ytrain,probs_train,labels=[0,1]),log_loss(ytest,probs_test,labels=[0,1])
    train_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (train_accuracy)
    print "Log Loss on training data: %0.2f" % (train_loss)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print "Log Loss on test data:     %0.2f" % (test_loss)
    return clf, Xtrain, ytrain, Xtest, ytest


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


if __name__=='__main__':
    S = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSSSS.SRRSSRSS;SSSS;RSRSRR;RSRSSS;SSSRS;SSRSS;SSSS;SSSRS;SSSRRRRSR.'
    #S = 'SSSS;RRRR;RSSSS;RSSRRSRSRR;SRRSSS;RRSSRR.SSRRSS;RSSSS;SRRRR;RSRRSR;SRSSRS;SSSRS;SSSS;RRRSSR;'
    S1 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R;'
    S2 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;'
    S3 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;S'
    S4 = 'SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSS'
    a,b,c,d = enumerate_pbp_2(S)




# def enumerate_pbp(s,partition):
#     sub_matches = ['']
#     if partition == 'set':
#         s = s.split('.')
#         for i in range(1,len(s)):
#             sub_matches.append('.'.join(s[:i])+'.')
#         return sub_matches[:-1]
    
#     elif partition=='game':
#         s = s.split(';')
#         s_new = []
#         for i in range(len(s)):
#             if '.' in s[i]:
#                 games = s[i].split('.')
#                 games[0] += '.'
#                 s_new += games
#             else:
#                 s_new += [s[i]]
#         for i in range(1,len(s_new)-1):
#             sub_matches.append((';'.join(s_new[:i])+';').replace('.;','.'))
#         return sub_matches[:-1]
    
#     # now, divide into points
#     elif partition=='point':
#         s = s.split(';')
#         s_new = []
#         for i in range(len(s)):
#             if '.' in s[i]:
#                 games = s[i].split('.')
#                 games[0] += '.'
#                 s_new += games
#             else:
#                 s_new += [s[i]]
#         for i in range(len(s_new)-1):
#             up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')
#             for k in range(len(s_new[i])):
#                 if s_new[i][k] not in ('.','/'):
#                     sub_matches.append(up_til_now+s_new[i][:k+1]+';' if k==len(s_new[i])-1 else up_til_now+s_new[i][:k+1])
#                 elif s_new[i][k]=='.':
#                     sub_matches[-1] += '.'
#                 #print 'sub: ', sub_matches
#         return sub_matches[:-1]

# v2.0, with completed sets
# def enumerate_pbp(s,partition):
#     sub_matches = ['']; sub_sets = [0]
#     completed_sets = 0
#     if partition == 'set':
#         s = s.split('.')
#         for i in range(1,len(s)):
#             sub_matches.append('.'.join(s[:i])+'.')
#         return sub_matches[:-1]
    
#     elif partition=='game':
#         s = s.split(';')
#         s_new = []
#         for i in range(len(s)):
#             if '.' in s[i]:
#                 games = s[i].split('.')
#                 games[0] += '.'
#                 s_new += games
#             else:
#                 s_new += [s[i]]
#         for i in range(1,len(s_new)-1):
#             sub_matches.append((';'.join(s_new[:i])+';').replace('.;','.'))
#         return sub_matches[:-1]
    
#     # now, divide into points
#     elif partition=='point':
#         s = s.split(';')
#         s_new = []
#         for i in range(len(s)):
#             if '.' in s[i]:
#                 games = s[i].split('.')
#                 games[0] += '.'
#                 s_new += games
#             else:
#                 s_new += [s[i]]
#         for i in range(len(s_new)-1):
#             up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')
#             for k in range(len(s_new[i])):
#                 if s_new[i][k] not in ('.','/'):
#                     sub_matches.append(up_til_now+s_new[i][:k+1]+';' if k==len(s_new[i])-1 else up_til_now+s_new[i][:k+1])
#                     sub_sets.append(completed_sets)
#                 elif s_new[i][k]=='.':
#                     sub_matches[-1] += '.'
#                     completed_sets+=1
#         return sub_matches[:-1],sub_sets[:-1]

'''
def enumerate_pbp_3(s,match_id,final_set_extend=0):

    # find the number of S,R,D,A characters and use this to initialize
    # all columns as npy arrays of this length
    length = len(s.replace('.','').replace('/','').replace(';',''))

    sub_matches = ['']; sub_sets = np.chararray(length)
    # obviously, would help further to make this into a matrix...
    all_scores = np.zeros([7,2,length])
    t_points = np.zeros(2)
    s_points = np.zeros(2)
    sw_points = np.zeros(2)
    points = np.zeros(2)
    games = np.zeros(2)
    sets = np.zeros(2)
    server = 0; current_ind=1


    t_points,p0_tp, p1_tp = np.zeros(2),np.zeros(length),np.zeros(length)
    s_points,p0_sp, p1_sp = np.zeros(2),np.zeros(length),np.zeros(length)
    sw_points,p0_swp, p1_swp = np.zeros(2),np.zeros(length),np.zeros(length)
    points,p0_p, p1_p = np.zeros(2),np.zeros(length),np.zeros(length)
    games,p0_g, p1_g = np.zeros(2),np.zeros(length),np.zeros(length)
    sets,p0_s, p1_s = np.zeros(2),np.zeros(length),np.zeros(length)
    server = 0; servers=np.zeros(length)
    current_ind=1
    
    # divides into s_new, array of games
    s = s.split(';'); s_new = []
    for i in range(len(s)):
        if '.' in s[i]:
            game_str = s[i].split('.')
            game_str[0] += '.'
            s_new += game_str
        else:
            s_new += [s[i]]

    # iterate through each game, a point at a time
    for i in range(len(s_new)-1):
        # update server; up_til_now is everything that has elapsed
        server = (server+1)%2 if i>0 else server
        up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')

        # update points and sets if there is a tiebreaker
        if games[0]==games[1]==6 and not (p0_s==p1_s==2 and final_set_extend):
            t_server = server
            mini_games = s_new[i].split('/')
            for j in range(len(mini_games)):
                for l in range(len(mini_games[j])):
                    if mini_games[j][l]!='.':
                        winner = int(mini_games[j][l] in ('S','A') and t_server==1 or mini_games[j][l] in ('R','D') and t_server==0) 
                        points[winner]+=1
                        t_points[winner]+=1
                        if winner==server: sw_points[winner]+=1
                        s_points[server]+=1
                        sub_m = up_til_now+'/'.join(mini_games[:j])+'/'+mini_games[j][:l+1] if j>0 else up_til_now+mini_games[j][:l+1]
                        sub_matches.append(sub_m)
                        p0_p.append(points[0]);p1_p.append(points[1])
                        p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                        p0_g.append(games[0]);p1_g.append(games[1])
                        p0_s.append(sets[0]);p1_s.append(sets[1])
                        p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                        p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                        servers.append(t_server)
                        current_ind+=1
                    else:
                        sets[winner]+=1
                        points, games = [0,0], [0,0]
                        p0_s[current_ind-1],p1_s[current_ind-1]=sets[0],sets[1]
                        p0_g[current_ind-1],p1_g[current_ind-1]=games[0],games[1]
                        p0_p[current_ind-1],p1_p[current_ind-1]=points[0],points[1]
                        sub_matches[current_ind-1] += '.'
                t_server = 1 - t_server

        # otherwise 
        else:
            for k in range(len(s_new[i])):
                if s_new[i][k]=='/':
                    print 'ERROR; this should be TIEBREAK'
                    return 0

                if s_new[i][k]!='.':    
                    winner = int(s_new[i][k] in ('S','A') and server==1 or s_new[i][k] in ('R','D') and server==0) 
                    points[winner]+=1
                    t_points[winner]+=1
                    if winner==server: sw_points[winner]+=1
                    s_points[server]+=1
                    if k==len(s_new[i])-1:
                        sub_matches.append(up_til_now+s_new[i][:k+1]+';')
                        games[winner]+=1
                        points = [0,0]
                    else:
                        sub_matches.append(up_til_now+s_new[i][:k+1])

                    p0_p.append(points[0]);p1_p.append(points[1])
                    p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                    p0_g.append(games[0]);p1_g.append(games[1])
                    p0_s.append(sets[0]);p1_s.append(sets[1])
                    p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                    p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                    servers.append(server)

                # backtrack and update previous entries if we finished a set; reset points and games
                elif s_new[i][k]=='.':
                    sets[winner]+=1
                    points, games = [0,0], [0,0]
                    p0_s[current_ind-1],p1_s[current_ind-1]=sets[0],sets[1]
                    p0_g[current_ind-1],p1_g[current_ind-1]=games[0],games[1]
                    p0_p[current_ind-1],p1_p[current_ind-1]=points[0],points[1]
                    sub_matches[current_ind-1] += '.'
        
        # reset points score
        points = [0,0]
        match_ids = [match_id]*len(servers)
        # have to add elo_diffs
    return [match_ids,sub_matches[:-1],servers,p0_s[:-1],p1_s[:-1],p0_g[:-1],p1_g[:-1],p0_p[:-1],p1_p[:-1],p0_tp[:-1],p1_tp[:-1],p0_swp[:-1],p0_sp[:-1],p1_swp[:-1],p1_sp[:-1]]
'''
