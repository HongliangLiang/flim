import os
# for i in range(58):
#     os.system('python3 run_classifier.py --model_type roberta --model_name_or_path microsoft/codebert-base --task_name codesearch --do_predict --output_dir ../data/codesearch/test/ --data_dir ../data/codesearch/test/ --max_seq_length 512 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 1e-5 --num_train_epochs 8 --test_file aspectj_'+str(i)+'.txt  --pred_model_dir ./models/java/checkpoint-best/ --test_result_dir ./results/java/batch_result_'+str(i)+'.txt')
from more_itertools import chunked
def calculate_same_value(labels_sorted, test_p_sorted, start_pos):
    i = start_pos
    num_same = 0
    num_p = 0
    while test_p_sorted[start_pos] == test_p_sorted[i]:
        num_same = num_same + 1
        if labels_sorted[i] ==1 : num_p = num_p + 1
        i = i + 1
        if i == len(labels_sorted ): break
    return num_p, num_same
def eval_mrr(test_p, labels):#在第二维相似度得分，真实标签
    test_p_sorted = test_p
    test_p_index = sorted(range(len(test_p_sorted)), key=lambda k: test_p_sorted[k], reverse=True)  # 降序排序
    test_p_sorted = sorted(test_p, reverse=True)

    labels_sorted = []
    for index in test_p_index:
        labels_sorted.append(labels[index])

    top_num = 10
    top10rank = 0
    for i in range(top_num):
        if (labels_sorted[i] == 1):
            '''
            num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted, i)
            num_r = top_num - i
            if num_p > (num_s-num_r):
                top10rank = 1
                break
            v1 = perm(num_s-num_r, num_p)*perm(num_s-num_p,num_s-num_p)
            v2 = perm(num_s,num_s)
            top10rank = 1-(float)((float)(v1)/(float)(v2))
            if top10rank > 1: top10rank=1
            if top10rank!=top10rank: top10rank=1
            break

    return top10rank

    '''
            top10rank = 1
            break
    num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted, 10)
    if (num_p >= 1): top10rank = 1  # 统计在第十位并列排名相同的文件中，是否含有相关文件

    MRRrank = 0.0
    for i in range(len(labels_sorted)):
        if (labels_sorted[i] == 1):
            MRRrank = MRRrank + float(1 / (i + 1))
            break

    MAPrank = 0.0
    pos_num = 0
    for i in range(len(labels_sorted)):
        if (labels_sorted[i] == 1):
            pos_num = pos_num + 1
            MAPrank = MAPrank + float(pos_num / (i + 1))
    if pos_num==0:
        print('出现不存在pos的例子')
        pos_num=1
    MAPrank = float(MAPrank / pos_num)
    # MRRrank = float(MRRrank / pos_num)

    return top10rank, MRRrank, MAPrank

#每1000验证一次三个指标
path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/0521_sumDes_tokSour_50.txt'
# path='/data/hdj/data/CodeBERT/codesearch/results/java/aspectj_withou_code_clasMeth_summDesc_result_all.txt'
# path='/data/hdj/data/CodeBERT/codesearch/results/java/swt_withou_code_clasMeth_summDesc_result_all.txt'
# path='/data/hdj/data/CodeBERT/codesearch/results/java/512_0202_result_all.txt'
with open(path,'r',encoding='utf-8') as f_in:
    data = f_in.readlines()
    print('读取的文件长度 :',len(data))
    batched_data = chunked(data, 50)
    print("start processing")
    top10ranks=[]
    MRRranks=[]
    MAPranks=[]
    for batch_idx, batch_data in enumerate(batched_data):
        preds=[]
        out_label_ids=[]
        report_ids=[]
        paths=[]
        for d_idx, d in enumerate(batch_data):
            line=d.split('<CODESPLIT>')
            out_label_ids.append(int(line[0]))
            preds.append(float(line[-1]))
            report_ids.append(line[1])
            paths.append(line[2])
        # print(len(preds),len(out_label_ids))
        top10rank, MRRrank, MAPrank = eval_mrr(preds, out_label_ids)
        print('top10rank, MRRrank, MAPrank',top10rank, MRRrank, MAPrank)
        top10ranks.append(top10rank)
        MRRranks.append(MRRrank)
        MAPranks.append(MAPrank)
        # with open('/data/hdj/data/CodeBERT/data/codesearch/test/aspectj_result_check/aspectj_'+str(batch_idx)+'.txt','w',encoding='utf-8') as f_out:
        #     # f_out.writelines('\n'.join(out_label_ids))
        #      for label,pred,id,path in zip(out_label_ids,preds,report_ids,paths):
        #          f_out.write(str(label)+" "+str(pred)+' '+id+' '+path+'\n')

    print('最后平均得分 top10rank, MRRrank, MAPrank :', sum(top10ranks)/len(top10ranks), sum(MRRranks)/len(MRRranks),sum(MAPranks)/len(MAPranks) )
