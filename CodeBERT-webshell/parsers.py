import glob
import os.path
from collections import OrderedDict

import xmltodict
import javalang
import pygments
from pygments.lexers import JavaLexer
from pygments.token import Token
import random
import numpy as np
import pickle
from collections import namedtuple
from pathlib import Path
class BugReport:
    """Class representing each bug report""" 
    
    __slots__ = ['summary', 'description', 'fixed_files',
                 'pos_tagged_summary', 'pos_tagged_description', 'stack_traces']
    
    def __init__(self, summary, description, fixed_files):
        self.summary = summary
        self.description = description
        self.fixed_files = fixed_files
        self.pos_tagged_summary = None
        self.pos_tagged_description = None
        self.stack_traces = None


class SourceFile:
    """Class representing each source file"""
    
    __slots__ = ['all_content', 'comments', 'class_names', 'attributes',
                 'method_names', 'variables', 'file_name', 'pos_tagged_comments',
                 'exact_file_name', 'package_name']
    
    def __init__(self, all_content, comments, class_names, attributes,
                 method_names, variables, file_name, package_name):
        self.all_content = all_content
        self.comments = comments
        self.class_names = class_names
        self.attributes = attributes
        self.method_names = method_names
        self.variables = variables
        self.file_name = file_name
        self.exact_file_name = file_name[0]
        self.package_name = package_name
        self.pos_tagged_comments = None


class Parser:
    """Class containing different parsers"""
    
    __slots__ = ['name', 'src', 'bug_repo']
    
    def __init__(self, project):
        self.name = project.name
        self.src = project.src
        self.bug_repo = project.bug_repo
        
    def report_parser(self):
        """Parse XML format bug reports"""
        
        # Convert XML bug repository to a dictionary
        with open(self.bug_repo,'rb') as xml_file:
        # with open('/home/hdj/bug_localization/data/AspectJ/AspectJBugRepository.xml','rb') as xml_file:
            xml_dict = xmltodict.parse(xml_file.read(), force_list={'file':True})
            
        # Iterate through bug reports and build their objects
        bug_reports = dict()

        for bug_report in xml_dict['bugrepository']['bug']:
            bug_reports[bug_report['@id']] = BugReport(
                bug_report['buginformation']['summary'],
                bug_report['buginformation']['description'] 
                    if bug_report['buginformation']['description'] else '',
                [os.path.normpath(path) for path in bug_report['fixedFiles']['file']]
            )
        
        return bug_reports

    def src_parser(self):
        """Parse source code directory of a program and collect
        its java files.
        """
        
        # Getting the list of source files recursively from the source directory
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)

        # Creating a java lexer instance for pygments.lex() method
        java_lexer = JavaLexer()
        
        src_files = OrderedDict()
        
        # Looping to parse each source file
        for src_file in src_addresses:
            with open(src_file,encoding='utf-8') as file:
                src = file.read()
                    
            # Placeholder for different parts of a source file
            comments = ''
            class_names = []
            attributes = []
            method_names = []
            variables = []

            # Source parsing
            parse_tree = None
            try:
                parse_tree = javalang.parse.parse(src)
                for path, node in parse_tree.filter(javalang.tree.VariableDeclarator):
                    if isinstance(path[-2], javalang.tree.FieldDeclaration):
                        attributes.append(node.name)
                    elif isinstance(path[-2], javalang.tree.VariableDeclaration):
                        variables.append(node.name)
            except:
                pass
            
            # Trimming the source file
            ind = False
            if parse_tree:
                if parse_tree.imports:
                    last_imp_path = parse_tree.imports[-1].path
                    src = src[src.index(last_imp_path) + len(last_imp_path) + 1:]
                elif parse_tree.package:
                    package_name = parse_tree.package.name
                    src = src[src.index(package_name) + len(package_name) + 1:]
                else:  # There is no import and no package declaration
                    ind = True
            # javalang can't parse the source file
            else:
                ind = True
            
            # Lexically tokenize the source file
            lexed_src = pygments.lex(src, java_lexer)
            
            for i, token in enumerate(lexed_src):
                if token[0] in Token.Comment:
                    if ind and i == 0 and token[0] is Token.Comment.Multiline:
                        src = src[src.index(token[1]) + len(token[1]):]
                        continue
                    comments += token[1]
                elif token[0] is Token.Name.Class:
                    class_names.append(token[1])
                elif token[0] is Token.Name.Function:
                    method_names.append(token[1])
            
            # Get the package declaration if exists
            if parse_tree and parse_tree.package:
                package_name = parse_tree.package.name
            else:
                package_name = None
            
            if self.name == 'aspectj':
                src_files[os.path.relpath(src_file, start=self.src)] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name
                )
            else:
                # If source file has package declaration
                if package_name:
                    src_id = (package_name + '.' + 
                              os.path.basename(src_file))
                else:
                    src_id = os.path.basename(src_file)
                        
                src_files[src_id] = SourceFile(
                    src, comments, class_names, attributes,
                    method_names, variables,
                    [os.path.basename(src_file).split('.')[0]],
                    package_name
                )
        print('here')
        # print(src_files.keys())
        print(type(src_files))
        f = open('out.pkl', 'wb')
        pickle.dump(src_files, f)
        return src_files
    def generate_triplet_data(self,x,an_pairs=10,testsize=0.3):
        triplet_train_pairs=[]
        triplet_test_pairs = []
        trainsize=1-testsize
        src_addresses = glob.glob(str(self.src) + '/**/*.java', recursive=True)
        for key,value in x.items():
            fixed_files=value.fixed_files
            # Getting the list of source files recursively from the source directory
            Neg_files=set(src_addresses)-set(fixed_files)
            A_P_len=len(fixed_files)
            Neg_files_k = random.sample(Neg_files, k=an_pairs)
            Neg_len=len(Neg_files_k)
            #train
            for pos_file in fixed_files[:max(int(A_P_len*trainsize),1)]:
                for neg_file in Neg_files_k:
                    # print('//////', key, pos_file, neg_file)
                    triplet_train_pairs.append([key,os.path.join(self.src,pos_file),neg_file])
             #test
            for pos_file in fixed_files[max(int(A_P_len*trainsize),1):]:
                for neg_file in Neg_files_k:
                    # print('*********', key, pos_file, neg_file)
                    triplet_test_pairs.append([key,os.path.join(self.src,pos_file),neg_file])
        return np.array(triplet_train_pairs),np.array(triplet_test_pairs)
def test():
    # import datasets

    _DATASET_ROOT = Path('/home/hdj/bug_localization/data')
    # _DATASET_ROOT = Path('/home/hdj/bug_localization/data')

    Dataset = namedtuple('Dataset', ['name', 'root', 'src', 'bug_repo'])

    # Source codes and bug repositories
    aspectj = Dataset(
        'aspectj',
        _DATASET_ROOT / 'AspectJ',
        _DATASET_ROOT / 'AspectJ/AspectJ-1.5',
        _DATASET_ROOT / 'AspectJ/AspectJBugRepository.xml'
    )
    # parser = Parser(datasets.aspectj)
    parser = Parser(aspectj)
    x = parser.report_parser()
    print(len(x))
    # d = parser.src_parser()
    
    # src_id, src = list(d.items())[10]
    # print('a :',src_id, src.exact_file_name, src.package_name)
    i=0
    # for key,value in x.items():
    #     i+=1
        # print(key,value.summary,value.description,value.fixed_files)

        # if i==10:
            # break
    # triplet_train_pair,triplet_test_pairs=parser.generate_triplet_data(x)
    # print(len(triplet_train_pair),len(triplet_test_pairs))
    src_files=parser.src_parser()
    for key,value in x.items():
        i+=1
        print(key,value)

        if i==10:
            break
    print(len(src_files))
import pandas as pd
if __name__ == '__main__':
    _DATASET_ROOT = Path('/home/hdj/bug_localization/data')
    Dataset = namedtuple('Dataset', ['name', 'root', 'src', 'bug_repo'])
    # Source codes and bug repositories
    aspectj = Dataset(
        'aspectj',
        _DATASET_ROOT / 'AspectJ',
        _DATASET_ROOT / 'AspectJ/AspectJ-1.5',
        _DATASET_ROOT / 'AspectJ/AspectJBugRepository.xml'
    )
    # parser = Parser(datasets.aspectj)
    parser = Parser(aspectj)
    x = parser.report_parser()
    print(len(x))
    i = 0
    for key,value in x.items():
        i+=1
        # print(key,value.summary,value.description,value.fixed_files)
        print(key)
        print(value.summary)
        print('*'*10)
        print(value.description)
        print('*' * 10)
        print(value.fixed_files)
        print('*' * 10)
        print(value.pos_tagged_summary)
        print('*' * 10)
        print(value.pos_tagged_description)
        print('*' * 10)
        print(value.stack_traces)
        print('*' * 10)
        if i==2:
            break
    # test()
    f=open('/home/hdj/bug_localization/bug-localization/buglocalizer/out.pkl', 'rb')
    data = pickle.load(f)
    dataFrame=pd.DataFrame()
    vals=[]
    i=0
    for key,value in data.items():
        # vals.append(value.all_content)
        i += 1


        # print(key, value.all_content,value.comments,value.class_names,value.attributes,value.method_names,
        #       value.variables,value.file_name,value.exact_file_name,value.package_name,value.pos_tagged_comments)
        print(key)
        print(value.all_content)#string public class 开始至最后
        print('*'*10)
        print(value.comments)#类型 string
        print('*' * 10)
        print(value.class_names)#类型 list ['Util', 'Constants', 'OSGIBundle', 'RequiredBundle']
        print('*' * 10)
        print(value.attributes)#list
        # ['TESTSRC', 'JAVA5_SRC', 'JAVA5_TESTSRC', 'JAVA5_VM', 'BUNDLE_NAME', 'BUNDLE_SYMBOLIC_NAME', 'BUNDLE_VERSION', 'BUNDLE_ACTIVATOR', 'BUNDLE_VENDOR', 'REQUIRE_BUNDLE', 'IMPORT_PACKAGE', 'BUNDLE_CLASSPATH', 'NAMES', 'manifest', 'attributes', 'text', 'name', 'versions', 'optional']
        print('*' * 10)
        print(value.method_names)#list
        # ['shortVersion', 'replace', 'visitFiles', 'iaxIfNotCanReadDir', 'iaxIfNotCanReadFile', 'iaxIfNotCanWriteDir', 'iaxIfNull', 'renderException', 'canWriteDir', 'path', 'path', 'canReadDir', 'canReadFile', 'delete', 'deleteContents', 'makeTempDir', 'close', 'isEmpty', 'closeSilently', 'closeSilently', 'reportMemberDiffs', 'copy', 'OSGIBundle', 'getAttribute', 'getClasspath', 'getRequiredBundles', 'RequiredBundle']
        print('*' * 10)
        print(value.variables)#list
        # ['java5VM', 'loc', 'result', 'start', 'files', 'passed', 'i', 'sw', 'pw', 'sb', 'i', 'files', 'i', 'tempFile', 'i', 'result', 'hits', 'i', 'curHit', 'j', 'prefix', 'i', 'i', 'result', 'names', 'cp', 'st', 'result', 'i', 'value', 'st', 'result', 'i', 'skips', 'token', 'first', 'patch', 'st', 'vers', 'opt', 'RESOLUTION', 'VERSION', 'token', 'start', 'end', 'start', 'end']
        print('*' * 10)
        print(value.file_name)#list ['Util']
        print('*' * 10)
        print(value.exact_file_name)#string Util
        print('*' * 10)
        print(value.package_name)#string org.aspectj.internal.tools.build
        print('*' * 10)
        print(value.pos_tagged_comments)
        print('*' * 10)

        # break
        if i == 2:
            break

    # dataFrame['src_path']=data.keys()
    # dataFrame['code']=vals
    # dataFrame.to_csv('data.csv',index=False)
    # print(data['org.aspectj/modules/testing-drivers/testdata/incremental/java/static/Target.40.java'].all_content)
    # print(str(data['org.aspectj/modules/testing-drivers/testdata/incremental/java/static/Target.40.java']))
