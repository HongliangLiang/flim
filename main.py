__author__ = 'Nick Hirakawa'

import pickle
from parse import *
from query import QueryProcessor
import operator


def main():
	# qp = QueryParser(filename='../text/queries.txt')
	# cp = CorpusParser(filename='../text/corpus.txt')
	# qp.parse()
	# queries = qp.get_queries()
	# cp.parse()
	# corpus = cp.get_corpus()
	queries=pickle.load(open('/data/hdj/SourceFile/data/sourceFile_eclipseUI/querys.pickle', 'rb'))
	corpus=pickle.load(open('/data/hdj/SourceFile/data/sourceFile_eclipseUI/corpus.pickle', 'rb'))
	print('queries ,corpus :',len(queries),len(corpus))
	proc = QueryProcessor(queries, corpus)
	results = proc.run()
	qid = 0
	for result in results:
		print('result len :',len(result))
		sorted_x = sorted(result.items(), key=operator.itemgetter(1))
		sorted_x.reverse()
		index = 0
		for i in sorted_x:
			tmp = (qid, i[0], index, i[1])
			print('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
			index += 1
		qid += 1


if __name__ == '__main__':
	main()
