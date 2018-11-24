# -*- coding: utf-8 -*-
import sys
import threading
import time
import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher

lucene.initVM(vmargs=['-Djava.awt.headless=true'])


class Ticker(object):
    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


class Indexer(object):
    """Usage: python IndexFiles <index_directory>"""

    def __init__(self, index_dir):
        print("lucene:", lucene.VERSION)
        self.index_dir = index_dir
        store = SimpleFSDirectory(Paths.get(self.index_dir))
        analyzer = LimitTokenCountAnalyzer(StandardAnalyzer(), 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(store, config)

    def build_index(self, id2content, id2title):
        print("loading data...")
        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        for k, v in id2content.items():
            doc = Document()
            doc.add(Field("id", k, t1))
            doc.add(Field("content", v, t2))
            doc.add(Field("title", id2title[k], t2))
            self.writer.addDocument(doc)

        ticker = Ticker()
        print("commit index")
        threading.Thread(target=ticker.run).start()
        self.writer.commit()
        self.writer.close()
        ticker.tick = False
        print("done")


class Queryer(object):
    def __init__(self, index_dir, top_k=5):
        self.directory = SimpleFSDirectory(Paths.get(index_dir))
        self.top_k = top_k
        self.searcher = IndexSearcher(DirectoryReader.open(self.directory))
        self.analyzer = StandardAnalyzer()

    def run_query(self, query):
        query = QueryParser("content", self.analyzer).parse(QueryParser.escape(query))
        scoreDocs = self.searcher.search(query, self.top_k).scoreDocs

        ids = []
        contents = []
        titles = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            ids.append(doc.get("id"))
            contents.append(doc.get("content"))
            titles.append(doc.get("title"))
        results = {"ids": ids,
                   "contents": contents,
                   "titles": titles}
        return results
