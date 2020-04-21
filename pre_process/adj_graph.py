def creat_sent_adj(vertex_num,none_label=None, knowledge_feature=None):
    adj_matrix = [[0] * vertex_num for _ in range(vertex_num)]
    adj_num=0
    pre = none_label
    for i,label in enumerate(knowledge_feature):
        if i==0:
            if label==none_label:
                pre=none_label
            else:
                pre=label
                adj_num+=1
        elif label!=none_label:
            if label == pre:
                adj_num +=1
            elif label!=pre and pre == none_label:
                adj_num =1
                pre = label
            elif label!=pre and pre!=none_label and adj_num>1:
                for j in range(1,adj_num):
                    x = i - j
                    for k in (1,adj_num):
                        y = i-k
                        if x == y :
                            continue
                        adj_matrix[x][y] =1
                        adj_matrix[y][x] =1
                adj_num =1
                pre = label
            elif label != pre and pre != none_label and adj_num == 1:
                adj_matrix[i-1][i-1] = 1
        elif label==none_label:
            if adj_num>1:
                for j in range(1,adj_num):
                    x = i - j
                    for k in (1,adj_num):
                        y = i-k
                        if x == y:
                            continue
                        adj_matrix[x][y] = 1
                        adj_matrix[y][x] = 1
            elif adj_num == 1:
                adj_matrix[i - 1][i - 1] = 1
            adj_num =0
            pre=label
        if i==len(knowledge_feature)-1  and adj_num>1:
            for j in range(1, adj_num):
                x = i - j+1
                for k in (1, adj_num):
                    y = i - k+1
                    if x == y :
                       continue
                    adj_matrix[x][y] = 1
                    adj_matrix[y][x] = 1
        elif i==len(knowledge_feature)-1 and adj_num == 1:
            adj_matrix[i][i] = 1
    return adj_matrix


if __name__ == '__main__':

    knowledge_feature =[[6,6, 10, 1, 1],[5, 5, 5, 2,10],[3, 3, 4, 4, 1]]

    for k in knowledge_feature:
        # print(k)
        adj=creat_sent_adj(5, none_label=10, knowledge_feature=k)
        for a in adj:
            print(a)

    # adj_mask = torch.gt(adj, 0)
    # adj = adj.masked_fill(adj_mask, 1)

