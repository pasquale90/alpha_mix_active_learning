import numpy as np
import torch



#cosine similarities
def sample_newMethod1( n, feats):
    '''
    input:
        n		: the number of samples requested
        feats	: the candidate feature embeddings
    
    N clusters are made. 
    The feature that is closer to the center of each cluster is selected.
    
    output:
        n number of samples out of the candidates.
    '''
    
    
    from scipy import sparse

    features=feats.numpy()

    A_sparse = sparse.csr_matrix(features)

    #similarity_matrix
    def similarity_matrix():
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(A_sparse)
        print('pairwise dense output:\n {}\n'.format(similarities))

        #also can output sparse matrices
        similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
        print('pairwise sparse output:\n {}\n'.format(similarities_sparse))
        
        # l2_nrm=torch.norm(features,dim=1,p=2)
        # l2_v1=torch.norm(features,dim=1)
        # l2_v2=torch.linalg.norm(features, dim=1, ord = 2)
        # l2_v3=features.pow(2).sum(dim=1).sqrt()
        return similarities
    
#distance matrix
# def distance_matrix():
    from sklearn.metrics import pairwise_distances

    distances=pairwise_distances(features,metric="euclidean") # metric="l2"
    # cosines= 
    # ang_similarities=similarity_matrix()
    # distances=distances*ang_similarities

    # np.fill_diagonal(distances, None)

    def get_connections():
        min_distance_from_sample=np.min(distances,axis=1,where=np.greater(distances,0.0),initial=distances[0,1])
        connections = [(i,np.where(distances[i]==min_distance_from_sample[i])[0][0]) for i in range(len(min_distance_from_sample))]
        # connected_nodes,min_distance_from_sample = get_connections()
        return connections,min_distance_from_sample
    def removeDuplicates(connections,minDists):
        nodubConnections=[]
        nodubMindistances=[]
        for i in range(len(connections)):
            c=connections[i]
            if ((c[0],c[1]) not in nodubConnections) and ((c[1],c[0]) not in nodubConnections): # remove dublicates
                nodubConnections.append(c)
                nodubMindistances.append(minDists[i])
        return nodubConnections,nodubMindistances

    def createGraph(connections):
        graph=[]
        for i in range(len(connections)):
            c=connections[i]
            n1=c[0]
            n2=c[1]

            # check if c1 and c2 are in graph
            def isingraph(node):
                for i in range(len(graph)):
                    l=graph[i]
                    for j in range(len(l)):
                        if l[j]==node:
                            # print(node," in ",i," ",graph)
                            return i
                # print(node,"not in ",graph)
                return -1

            idx1=isingraph(n1)
            idx2=isingraph(n2)

            
            if idx1==-1 and idx2==-1:
                newteam=[]
                newteam.append(n1)
                newteam.append(n2)
                graph.append(newteam)
            elif idx1!=-1 and idx2!=-1:
                if (idx1!=idx2):        # two different teams are connected
                    #concatenate teams
                    graph[idx1]=graph[idx1]+graph[idx2]
                    # remove copied team
                    graph.pop(idx2)
                else:                   # already in the same team
                    pass
            elif idx1!=-1:
                # print("appending ",n2," to idx ",idx1,"-->",graph[idx1])
                graph[idx1].append(n2)
            elif idx2!=-1:
                # print("appending ",n1," to idx ",idx2,"-->",graph[idx2])
                graph[idx2].append(n1)

        return graph

    def selectPopularSamples(pwdists,conns,dists,n):
        counts=np.zeros(n)
        distsum=np.zeros(n)
        graph=[]
        
        for i in range(len(conns)):
            c=conns[i]
            n1=c[0]
            n2=c[1]
            d=dists[i]

            counts[n2]+=1
            distsum[n2]+=d
            # counts[n1]+=1
            # distsum[n1]+=d

            # check if c1 and c2 are in graph
            def isingraph(node):
                for i in range(len(graph)):
                    l=graph[i]
                    for j in range(len(l)):
                        if l[j]==node:
                            # print(node," in ",i," ",graph)
                            return i
                # print(node,"not in ",graph)
                return -1

            idx1=isingraph(n1)
            idx2=isingraph(n2)

            
            if idx1==-1 and idx2==-1:
                newteam=[]
                newteam.append(n1)
                newteam.append(n2)
                graph.append(newteam)
                
            elif idx1!=-1 and idx2!=-1:
                if (idx1!=idx2):        # two different teams are connected
                    #concatenate teams
                    graph[idx1]=graph[idx1]+graph[idx2]
                    # remove copied team
                    graph.pop(idx2)
                else:                   # already in the same team
                    pass
            elif idx1!=-1:
                # print("appending ",n2," to idx ",idx1,"-->",graph[idx1])
                graph[idx1].append(n2)
            elif idx2!=-1:
                # print("appending ",n1," to idx ",idx2,"-->",graph[idx2])
                graph[idx2].append(n1)

        distmean=distsum/counts
        max_idxs=np.argsort(counts)[::-1] # samples that are close to as many other samples (max_other_samples)
        mindist_idxs=np.argsort(distmean) # samples that has the minimum 

        lenSubCl=[len(l) for l in graph]
        meanLen=np.mean(lenSubCl)
        stdLen=np.std(lenSubCl)
        medianLen=sorted([len(l) for l in graph])[int(len(graph)/2)]
        filtered_mean=[len(l) for l in graph if len(l)>meanLen]
        filtered_median=[len(l) for l in graph if len(l)>medianLen]

        maxLenTeams=np.argsort(lenSubCl)[::-1]

        final_idxs=[]
        final_dists=[]
        for team in maxLenTeams[:100]:
            currTeam=graph[team]
            # for sample in max_idxs:
            mean_team_distances=np.zeros(len(currTeam)) #the mean distance of_each_sample to all the others in the same team
            for s in range(len(currTeam)):
                mean_dist=0
                currSample=currTeam[s]
                for ex in range(len(currTeam)):
                    if ex == s :
                        pass
                    else:
                        currComp=currTeam[ex]
                        currDist=pwdists[currSample][currComp]
                        assert currDist==pwdists[currComp][currSample]
                        # print(f'team {team} : comparing {s}({currSample}) with {ex}({currTeam[ex]})')
                        mean_dist+=currDist
                # mean_team_distances[s]=mean_dist/len(graph[team])
                mean_team_distances[s]=mean_dist/len(currTeam)
                # import pdb
                # pdb.set_trace()
            mostPopSample_idx=np.argsort(mean_team_distances)[0]
            mostPopSample=currTeam[mostPopSample_idx]
            final_idxs.append(mostPopSample)

        return np.array(final_idxs)
    
    connected_nodes,min_distance_from_sample = get_connections()
    nodubConnections,nodubMindistances=removeDuplicates(connected_nodes,min_distance_from_sample)
    return selectPopularSamples(distances,nodubConnections,nodubMindistances,len(feats))