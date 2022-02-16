import networkx as nx
import math
import operator
import matplotlib.pyplot as plp
import pandas as pd
import numpy as np

'''
MCDE(v) = core(v) + degree(v) + entropy(v)
pi = (v's neighbors occur in ith shell) / d_v
entropy = -sum_i=0^max(shell) pi*log2(po)
'''
def get_weight(G, degree_dic, rank):  # wij表示：i给j投票的能力，wij ！= wji, rank is seed noods
    weight = {}
    nodes = nx.nodes(G)
    rank_list = [i[0] for i in rank]
    for node in nodes:
        sum1 = 0
        neighbors = list(nx.neighbors(G, node))
        neighbors_common_rank = list(set(neighbors) & set(rank_list))
        if len(neighbors_common_rank) != 0:  # 节点与已选节点的直接为0
            for nc in neighbors_common_rank:
                weight[(node, nc)] = 0
        neighbours_without_rank = list(set(neighbors) - set(rank_list))  # voting for unselected nodes
        if len(neighbours_without_rank) != 0:  # if the node has other nieghbours
            for nbr in neighbours_without_rank:
                sum1 += degree_dic[nbr]
            for neigh in neighbours_without_rank:
                weight[(node, neigh)] = degree_dic[neigh] / sum1
        else:  # 当前节点只有已选节点作为邻居
            for neigh in neighbors:
                weight[(node, neigh)] = 0
    # for i, j in nx.edges(G):
    #     sum1 = 0
    #     for nbr in nx.neighbors(G, i):
    #         sum1 += degree_dic[nbr]
    #     weight[(i, j)] = degree_dic[j] / sum1
    return weight

def get_node_score2(G, nodesNeedcalcu, node_ability, degree_dic, rank):

    weight = get_weight(G, degree_dic, rank)
    node_score = {}
    for node in nodesNeedcalcu:  # for ever node add the neighbor's weighted ability
        sum2 = 0
        neighbors = list(nx.neighbors(G, node))
        for nbr in neighbors:
            sum2 += 0.5*node_ability[nbr] * (weight[(nbr, node)] + (G[nbr][node]['weight']) *len(neighbors) / degree_dic[node])
        node_score[node] = math.sqrt(len(neighbors) * sum2)
    return node_score




def mean_value(a):
    n = len(a)
    sum_mean = 0
    for i in a:
        sum_mean += i
    return sum_mean / n
def get_keys(d, va):
    for k, v in d.items():
        if va in v:
            return k
def next_f(value, lis):
    '''
    :param value: the value needed to compare
    :param lis: list
    :return: 返回列表lis中第一个不等于value的元素下标，如果都等于，则默认返回第一个元素下标
    '''

    for i in lis:
        if i != value:
            return lis.index(i)
    return 0
def k_shell(graph):
    importance_dict = {}
    level = 1
    while len(graph.degree):
        importance_dict[level] = []
        while True:
            level_node_list = []
            for item in graph.degree:
                if item[1] <= level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree,key=lambda x: x[1])[1] > level:
                break
        level = min(graph.degree,key=lambda x:x[1])[1]
    return importance_dict
def HWVoteRank(G, l, lambdaa,beta):
    '''

    :param G: use new indicator + lambda + voterank, the vote ability = log(dij)
    :param l: the number of spreaders
    :param lambdaa: retard infactor
    :return:
    '''
    rank = []

    nodes = list(nx.nodes(G))
    

    
    degree_li = nx.degree(G,weight="weight")
    d_max = max([i[1] for i in degree_li])
    
    degree_dic = {}
    for i in degree_li:
        degree_dic[i[0]] = i[1]

    node_ability = {}
    for item in degree_li:
        node_ability[item[0]] = 1  # ln(x)


    
    degree_values = degree_dic.values()
    weaky = 1 / mean_value(degree_values)
    # node's score
    node_score = get_node_score2(G, nodes, node_ability, degree_dic, rank)
    

    node_type = pd.DataFrame()
    node_type['nodes'] = nodes
    node_type.set_index(["nodes"], inplace=True)
    node_type ['degree'] = [0] * node_type.shape[0]
    node_type ['lable'] = [5] * node_type.shape[0]
    
    

    node_type_all = pd.read_csv('./degree_lable.csv')
    node_type_all.set_index(["gene"], inplace=True)
    for ii in list(node_type.index.values):
        node_type.loc[ii,'degree'] = degree_dic[ii]
        node_type.loc[ii,'lable'] = node_type_all.loc[ii,'lable']
        
    node_mean_TypeDegree = node_type.groupby('lable')['degree'].mean()
    
    for ii in list(node_type.index.values):
        if node_type.loc[ii,'lable'] == 0:
            node_type.loc[ii,'degree'] = node_mean_TypeDegree [0]
        if node_type.loc[ii,'lable'] == 1:
            node_type.loc[ii,'degree'] = node_mean_TypeDegree [1]
        if node_type.loc[ii,'lable'] == 2:
            node_type.loc[ii,'degree'] = node_mean_TypeDegree [2]

    for i in range(l):
        max_score_node, score = max(node_score.items(), key=lambda x: x[1])
        rank.append((max_score_node, score))
        node_ability[max_score_node] = 0
        node_score.pop(max_score_node)
        # for the max score node's neighbor conduct a neighbour ability surpassing
        cur_nbrs = list(nx.neighbors(G, rank[-1][0]))
        # spreader's neighbour 1 th neighbors
        
        next_cur_neigh = []# spreader's neighbour's neighbour 2 th neighbors
        #next_cur_neigh_type = []
        next_cur_neigh_1 = []
        for nbr in cur_nbrs:
            nnbr = nx.neighbors(G, nbr)
            #next_cur_neigh_1.extend(nnbr)
            for node_nnbr in nnbr:
                if node_type.loc[node_nnbr,'lable'] == node_type.loc[nbr,'lable'] and node_type.loc[max_score_node,'lable'] == node_type.loc[nbr,'lable']:
                    next_cur_neigh.append(node_nnbr)
            
            
            if node_type.loc[max_score_node,'lable'] == node_type.loc[nbr,'lable']:
                #node_ability[nbr] *= lambdaa  # suppress the 1th neighbors' voting ability
                next_cur_neigh_1.extend(nx.neighbors(G, nbr))
                node_ability[nbr] = node_ability[nbr] - 1 / node_type.loc[nbr,'degree'] #weaky
                if node_ability[nbr] < 0:
                    node_ability[nbr] = 0

        next_cur_neighs = list(set(next_cur_neigh))  # delete the spreaders and the 1th neighbors
        for ih in rank:
            if ih[0] in next_cur_neighs:
                next_cur_neighs.remove(ih[0])
        for i in cur_nbrs:
            if i in next_cur_neighs:
                next_cur_neighs.remove(i)

        for nnbr in next_cur_neighs:
            
            node_ability[nnbr] =  node_ability[nnbr] - 1 / (2 *  node_type.loc[nnbr,'degree']) #weaky)
            if node_ability[nnbr] < 0:
                node_ability[nnbr]=0
            
        next_cur_neighs_1 = list(set(next_cur_neigh_1))  # delete the spreaders and the 1th neighbors
        for jh in rank:
            if jh[0] in next_cur_neighs_1:
                next_cur_neighs_1.remove(jh[0])
        for j in cur_nbrs:
            if j in next_cur_neighs_1:
                next_cur_neighs_1.remove(j)
        
        
        
        # find the neighbor and neighbor's neighbor
        H = []
        H.extend(cur_nbrs)
        H.extend(next_cur_neighs_1)
        for nbr in next_cur_neighs:
            nbrs = nx.neighbors(G, nbr)
            H.extend(nbrs)

        H = list(set(H))  ##the 1th neighbors
        for ih in rank:
            if ih[0] in H:
                H.remove(ih[0])
        new_nodeScore = get_node_score2(G, H, node_ability, degree_dic, rank)
        node_score.update(new_nodeScore)

 
    return rank

