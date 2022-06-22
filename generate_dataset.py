#获取bug_report TODO
import json
def clean_string(string):
    outstring_list=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',string)
    return ' '.join(outstring_list)
# report_id_list,source_id_list
def get_method_top_k(choosed_methods:list,k=50):
    method_list=[]
    for string_bef in choosed_methods:
#         print('string_bef :',string_bef)
        string=clean_string_code(string_bef)
        if '{'  in string and '}'in string :
            method_list.append(string)
#     print('函数体的方法数量: ',len(method_list))
    num=len(method_list)
    method_list = sorted(method_list,key = lambda string:len(string.split()),reverse=True)
    if len(method_list)==0:
        for string_bef in choosed_methods:
#         print('string_bef :',string_bef)
            string=clean_string(string_bef)
            method_list.append(string)
        method_list = sorted(method_list,key = lambda string:len(string.split()),reverse=True)
        num=len(method_list)
        return method_list[:k],num
#     print(len(method_list))
#     for method in method_list:
#         print(method)
    return method_list[:k],num

def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports

# bug_report_file_path='/data/hdj/tracking_buggy_files/'+swt+'.json'
# project='aspectj'
# project='tomcat'
project='swt'
bug_report_file_path='/data/hdj/tracking_buggy_files/'+project+'/'+project+'.json'
bug_reports = load_bug_reports(bug_report_file_path)
#获取code信息
# data_prefix='/data/hdj/tracking_buggy_files/tomcat/tomcat'
# ast_cache_db = UnQLite(data_prefix+"_ast_cache_collection_db")
# project='eclipse_platform_ui'

ast_cache_collection_db=UnQLite("/data/hdj/tracking_buggy_files/"+project+'/'+project+"_ast_cache_collection_db",
                                             flags=0x00000100 | 0x00000001)
ast=[item for item in ast_cache_collection_db]
def clean_string_code(string):
    m = re.compile(r'/\*.*?\*/', re.S)
    outstring = re.sub(m, '', string)
    m = re.compile(r'import*', re.S)
    outstring = re.sub(m, '', outstring)
    m = re.compile(r'//.*')
    outtmp = re.sub(m, '', outstring)
    for char in ['\r\n', '\r', '\n']:
        outtmp = outtmp.replace(char, ' ')
    outtmp=' '.join(outtmp.split())
    outtmp=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',outtmp)
    return ' '.join(outtmp)
# for key,val in tomcat_ast_file.items():
#     print(key,type(val),len(val))


def generate_pli(train_k_path,out_path):
    #生成PLI需要的数据集
    task1_list=[]
    #train_k_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/train_1.csv' 
    #out_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/train_1.txt'
    train_k=pd.read_csv(train_k_path)
    train_k.set_index(['bid','fid'], inplace = True)
    train_k.head(10)
    #取bid所对应的report
    bid_list=list(train_k.index.get_level_values(0).unique())
    len(bid_list)
    summarys=[]
    descriptions=[]
    for bug_report_id in bid_list:
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        summarys.append(current_bug_report['summary'])
        descriptions.append(current_bug_report['description'])
    report_dataFrame=pd.DataFrame({'summary':summarys,'description':descriptions},index=bid_list)
    report_dataFrame.index.names=['bid']
    def remove_twoHeadWord(string):
        contents=string.split(' ')[2:]
        return ' '.join(contents)
    report_dataFrame.fillna("",inplace=True)
    report_dataFrame['summary']=report_dataFrame['summary'].apply(remove_twoHeadWord)
    report_dataFrame['summary']=report_dataFrame['summary'].apply(clean_string_report)
    report_dataFrame['description']=report_dataFrame['description'].apply(clean_string_report)
    report_dataFrame.head()
    all_ast_index=list(train_k.index.get_level_values(1).unique())
    type(all_ast_index)
#     all_ast_file_tokenizedSource=[]
    all_ast_file_tokenizedMethods=[]
    all_ast_file_tokenizedClassNames =[]
    all_ast_file_tokenizedMethodNames=[]
    all_ast_file_tokenizedVariableNames=[]
    all_ast_file_tokenizedComments =[]
    all_ast_file_methods=[]
    for ast_index in all_ast_index:
    #     print(ast_index)
        ast_file = pickle.loads(swt_ast_cache_collection_db[ast_index])
#         all_ast_file_tokenizedSource.append(convert_dict2string_set([ast_file['tokenizedSource']]))
        all_ast_file_tokenizedMethods.append(convert_dict2string_set(ast_file['tokenizedMethods']))
        top_k_method,num=get_method_top_k(ast_file['methodContent'],10)
        print('total method num :',len(ast_file['methodContent']),' 有方法体的函数数量 :',num,' choosed method num :',len(top_k_method))
        all_ast_file_methods.append(top_k_method )
        all_ast_file_tokenizedClassNames.append(convert_dict2string_set(ast_file['tokenizedClassNames']))
        all_ast_file_tokenizedMethodNames.append(convert_dict2string_set(ast_file['tokenizedMethodNames']))
        all_ast_file_tokenizedVariableNames.append(convert_dict2string_set(ast_file['tokenizedVariableNames']))
        all_ast_file_tokenizedComments.append(convert_dict2string_set(ast_file['tokenizedComments']))
    # all_ast_index
    all_ast_index_dataframe=pd.DataFrame({'all_ast_file_methods':all_ast_file_methods,'tokenizedMethods':all_ast_file_tokenizedMethods,
                                          'tokenizedClassNames':all_ast_file_tokenizedClassNames,'tokenizedMethodNames':all_ast_file_tokenizedMethodNames,
                                          'tokenizedVariableNames':all_ast_file_tokenizedVariableNames,'tokenizedComments':all_ast_file_tokenizedComments},index=all_ast_index)
    # print(all_ast_index_dataframe.head())
    print(all_ast_index_dataframe.shape)
    # all_ast_index_dataframe.set_index(["ast_index"], inplace=True)
    # all_ast_index_dataframe
    all_ast_index_dataframe.index.names=['fid']
    all_ast_index_dataframe.head(2)
    #开始连接字段
    all_methodContent_index_dataframe=all_ast_index_dataframe.join(train_k,how='inner')
    all_methodContent_index_dataframe.head()
    all_report_methodContent_index_dataframe=report_dataFrame.join(all_methodContent_index_dataframe,how='inner')
    all_report_methodContent_index_dataframe.head()
    
    task1_list=[]
    all_report_methodContent_index_dataframe.fillna("",inplace=True)
    sum_label=0
    for i,(index, row) in enumerate(all_report_methodContent_index_dataframe.iterrows()):
    #     print(index[0],index[1],row['summary'],row['description'],row['methodContent'],row['used_in_fix'])
#         if len(row['tokenizedSource'])==0:
#             continue
        guid='_'.join([index[0],index[1]])
        
#         q_paras=[mergeTokenziedSource(tokenize(row['summary'],stemmer)),mergeTokenziedSource(tokenize(row['description'],stemmer))]
        q_paras=[row['summary'],row['description']]
        print('tokenizedMethods len :',len(row['tokenizedMethods']))
        c_paras=[row['tokenizedMethods'],row['tokenizedClassNames'],
                 row['tokenizedMethodNames'],row['tokenizedVariableNames'],row['tokenizedComments'],]
#         print('all_ast_file_methods :',len(row['all_ast_file_methods']))
        c_paras.extend(row['all_ast_file_methods'])
#         c_paras=row['all_ast_file_methods']
#         print(i,'summary :',type(row['summary']),'description :',type(row['description']),'len summary :',len(q_paras.split()),'source :',len(c_paras.split()))
        summary_len.append(len(q_paras[0].split()))
        description_len.append(len(q_paras[1].split()))
#         tokenizedSource_len.append(len(c_paras[0].split()))

#         tokenizedMethods_len.append(len(c_paras[1].split()))      
#         tokenizedClassNames_len.append(len(c_paras[2].split()))   
#         tokenizedMethodNames_len.append(len(c_paras[3].split()))   
#         tokenizedVariableNames_len.append(len(c_paras[4].split())) 
#         tokenizedComments_len.append(len(c_paras[5].split()))
        label=int(row['used_in_fix'])
        task1=dict()
        task1['guid']=guid
        task1['q_paras']=q_paras
        task1['c_paras']=c_paras
        task1['label']=label
        task1_list.append(task1)
#         if i==10:
#             break
    print(task1_list[0])
    with open(out_path,'w',encoding='utf-8') as f_out:
        for task1 in task1_list:
            out_line = json.dumps(task1, ensure_ascii=False) + '\n'
            f_out.write(out_line)
summary_len=[]
description_len=[]
tokenizedSource_len=[]
tokenizedMethods_len=[]
tokenizedClassNames_len=[]
tokenizedMethodNames_len=[]
tokenizedVariableNames_len=[]
tokenizedComments_len=[]
train_k_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/train_3hard_1.csv' 
train_out_path='/data/hdj/BERT_PLI/train_3hard_1_5_10method.json' 
generate_pli(train_k_path,train_out_path)

test_2_k_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/test_3hard_2.csv' 
test_2_out_path='/data/hdj/BERT_PLI/test_3hard_2_5_10method.json'  
generate_pli(test_2_k_path,test_2_out_path)

test_k_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/test_half_hardeasy_3.csv' 
test_out_path='/data/hdj/BERT_PLI/test_half_hardeasy_3.json'  
# generate_pli(test_k_path,test_out_path)


def generate_fine_tune(train_k_path,out_path):
    #生成fine-tune需要的数据集
    #train_k_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/train_1.csv' 
    #out_path='/home/hdj/BERT-PLI-IJCAI2020/examples/task2/data_sample_1.json'
    train_k=pd.read_csv(train_k_path)
    train_k.set_index(['bid','fid'], inplace = True)
    train_k.head(10)
    #取bid所对应的report
    bid_list=list(train_k.index.get_level_values(0).unique())
    len(bid_list)
    summarys=[]
    descriptions=[]
    for bug_report_id in bid_list:
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        summarys.append(current_bug_report['summary'])
        descriptions.append(current_bug_report['description'])
    report_dataFrame=pd.DataFrame({'summary':summarys,'description':descriptions},index=bid_list)
    report_dataFrame.index.names=['bid']
    def remove_twoHeadWord(string):
        contents=string.split(' ')[2:]
        return ' '.join(contents)
    report_dataFrame['summary']=report_dataFrame['summary'].apply(remove_twoHeadWord)
    report_dataFrame.head()
    all_ast_index=list(train_k.index.get_level_values(1).unique())
    type(all_ast_index)
    all_ast_file_methodContent=[]
    for ast_index in all_ast_index:
    #     print(ast_index)
        ast_file = pickle.loads(swt_ast_cache_collection_db[ast_index])
        all_ast_file_methodContent.append(ast_file['methodContent'])
    # all_ast_index
    all_ast_index_dataframe=pd.DataFrame({'methodContent':all_ast_file_methodContent},index=all_ast_index)
    # print(all_ast_index_dataframe.head())
    print(all_ast_index_dataframe.shape)
    # all_ast_index_dataframe.set_index(["ast_index"], inplace=True)
    # all_ast_index_dataframe
    all_ast_index_dataframe.index.names=['fid']
    all_ast_index_dataframe.head(2)
    #开始连接字段
    all_methodContent_index_dataframe=all_ast_index_dataframe.join(train_k,how='inner')
    all_methodContent_index_dataframe.head()
    all_report_methodContent_index_dataframe=report_dataFrame.join(all_methodContent_index_dataframe,how='inner')
    all_report_methodContent_index_dataframe.head()
    
    task1_list=[]
    all_report_methodContent_index_dataframe.fillna("",inplace=True)
    def clean_string(string):
        outstring_list=re.findall(r'[\w"]+|[.,!?;{}:()\+\-\*\/=><"]',string)
        return ' '.join(outstring_list)
    sum_label=0
    for i,(index, row) in enumerate(all_report_methodContent_index_dataframe.iterrows()):
    #     print(index[0],index[1],row['summary'],row['description'],row['methodContent'],row['used_in_fix'])
        if len(row['methodContent'])==0:
            continue
        guid='_'.join([index[0],index[1]])
#         print(i,'summary :',type(row['summary']),'description :',type(row['description']))
#         q_paras=[clean_string(row['summary']),clean_string(row['description'])]
#         c_paras=[clean_string(string) for string in row['methodContent']]
        q_paras=[mergeTokenziedSource(tokenize(row['summary'],stemmer)),mergeTokenziedSource(tokenize(row['description'],stemmer))]
        c_paras=[mergeTokenziedSource(tokenize(string,stemmer)) for string in row['methodContent']]
        
        label=int(row['used_in_fix'])

        for q in q_paras:
            if q=="":
                continue
            for c in c_paras:
                if c=="":
                    continue
                task1=dict()

                task1['guid']=guid
                task1['text_a']=q
                task1['text_b']=c
                task1['label']=label
                sum_label+=label
                task1_list.append(task1)
    #     print(task1)
    #     if i==2:
    #         break
    print(len(task1_list),sum_label,sum_label/len(task1_list))
    task1_list[0]
    label=0
    label_0=0
    for task in task1_list:
    #     print(task)
        label+=task['label']
        if task['label']==0:
            label_0+=1
    #     break
    print(label)
    with open(out_path,'w',encoding='utf-8') as f_out:
        for task1 in task1_list:
            out_line = json.dumps(task1, ensure_ascii=False) + '\n'
            f_out.write(out_line)
train_k_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/train_1.csv'
out_path='/data/hdj/tracking_buggy_files/joblib_memmap_swt/train_1_fine_tune.txt'
generate_fine_tune(train_k_path,out_path)