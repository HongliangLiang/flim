
from unqlite import UnQLite
import pickle


#将swt的两个ast_cache合并
project='jdt'
ast_raw_cache_collection_db=UnQLite("/data/hdj/tracking_buggy_files/"+project+'/'+project+"_raw_ast_cache_collection_db",flags=0x00000100 | 0x00000001)
ast_raw=[item for item in ast_raw_cache_collection_db]

#合并结束
ast_tokenzized_cache_collection_db=UnQLite("/data/hdj/tracking_buggy_files/"+project+'/'+project+"_tokenized_ast_cache_collection_db",flags=0x00000100 | 0x00000001)

ast_tokenzized=[item for item in ast_tokenzized_cache_collection_db]
# ast_cache=dict()
ast_cache_collection_db = UnQLite('/data/hdj/tracking_buggy_files/'+project+'/'+project+'_ast_cache_collection_db')
for item in ast_raw:
    sha=item[0]
    ast_raw_file = pickle.loads(ast_raw_cache_collection_db[sha])
    ast_tokenzized_file = pickle.loads(ast_tokenzized_cache_collection_db[sha])
    ast_tokenzized_file['methodContent']=ast_raw_file['methodContent']
    ast_tokenzized_file['rawSourceContent']=ast_raw_file['rawSourceContent']
    ast_tokenzized_file['commentContent']=ast_raw_file['commentContent']
    ast_tokenzized_file['methodNames']=ast_raw_file['methodNames']
    # ast_cache[sha]=ast_tokenzized_file
    ast_cache_collection_db[sha] = pickle.dumps(ast_tokenzized_file, -1)

# 将ast解析的信息进行保存
# for k, v in ast_cache.items():
