from database_creation.utils import BaseClass, Tuple, Wikipedia, Query
from database_creation.article import Article

from numpy.random import shuffle, seed
from glob import glob
from collections import defaultdict
from numpy import histogram
from pickle import dump, load, PicklingError
from pandas import DataFrame

import matplotlib.pyplot as plt


class Database(BaseClass):
    # region Class initialization

    to_print, print_attributes, print_lines, print_offsets = ['articles'], False, 2, 0
    modulo_articles, modulo_tuples, modulo_entities = 1000, 500, 1000

    def __init__(self, years=(2006, 2007), max_size=None, project_root='', verbose=True, min_articles=None,
                 min_queries=None):
        """
        Initializes an instance of Database.

        Args:
            years: list, years (int) of the database to analyse.
            max_size: int, maximum number of articles in the database; if None, takes all articles.
            project_root: str, relative path to the root of the project.
            verbose: bool, verbose option of the database.
            min_articles: int, minimum number of articles an entities' tuple must be in.
            min_queries: int, minimum number of Queries an entities' tuple must have.
        """

        self.years = years
        self.max_size = max_size
        self.project_root = project_root
        self.verbose = verbose
        self.min_articles = min_articles
        self.min_queries = min_queries

        self.articles = None
        self.entities = None
        self.tuples = None
        self.wikipedia = None
        self.queries = None
        self.stats = None

    def __str__(self):
        """
        Overrides the builtin str method, customized for the instances of Database.

        Returns:
            str, readable format of the instance.
        """

        to_print, print_attribute, print_lines, print_offsets = self.get_parameters()[:4]
        random_print, limit_print = False, 20
        attributes = to_print or list(self.__dict__.keys())

        string = ''

        for attribute in attributes:
            s = self.to_string(getattr(self, attribute)) if attribute != 'articles' else ''
            string += self.prefix(print_attribute, print_lines if string else 0, print_offsets, attribute) + s if s \
                else ''

        if 'articles' in attributes:
            ids = set(self.articles.keys())
            if random_print:
                shuffle(ids)

            string += self.prefix(True, print_lines if string else 0, print_offsets, 'articles') \
                if print_attribute else ''

            count = 0
            for id_ in ids:
                s = self.to_string(self.articles[id_])

                if s:
                    string += self.prefix(print_attribute, print_lines if string else 0, print_offsets,
                                          'article ' + str(id_)) + s
                    count += 1
                    if count == limit_print:
                        break

        return string

    # endregion

    # region Main methods

    @BaseClass.Verbose("Preprocessing the database...")
    def preprocess_database(self):
        """ Performs the preprocessing of the database. """

        self.compute_articles()

        self.clean_articles(criterion=Article.criterion_content)
        self.clean_articles(criterion=Article.criterion_summary)

        self.compute_metadata()
        self.compute_entities()
        self.compute_tuples()

        self.filter(min_articles=self.min_articles)

    @BaseClass.Verbose("Preprocessing the articles...")
    def process_articles(self):
        """ Performs the preprocessing of the articles. """

        self.compute_annotations()
        self.compute_contexts()

        self.filter(min_queries=self.min_queries)

    @BaseClass.Verbose("Processing the wikipedia information...")
    def process_wikipedia(self, load=False, file_name=None):
        """
        Performs the processing of the wikipedia information of the database.

        Args:
            load: bool, if True, load an existing file.
            file_name: str, name of the wikipedia file to load; if None, load the standard files.
        """

        if load:
            self.load_pkl(attribute_name='wikipedia', file_name=file_name, folder_name='wikipedia')
            self.compute_wikipedia(load=load)

        else:
            self.compute_wikipedia(load=load)
            self.save_pkl(attribute_name='wikipedia', folder_name='wikipedia')

    @BaseClass.Verbose("Processing the aggregation queries...")
    def process_queries(self, load=False, file_name=None):
        """
        Performs the processing of the aggregation queries.

        Args:
            load: bool, if True, load an existing file.
            file_name: str, name of the queries file to load; if None, load the standard file.
        """

        if load:
            self.load_pkl(attribute_name='queries', file_name=file_name)

        else:
            self.compute_queries()
            self.save_pkl('queries')
            self.save_csv(attribute_name='queries', limit=200)

    @BaseClass.Verbose("Computing and displaying statistics...")
    def process_stats(self, type_):
        """
        Compute and display the statistics of the database of the given type.

        Args:
            type_: str, type of the statistics, must be 'tuples', 'wikipedia' or 'contexts'.
        """

        getattr(self, 'compute_stats_' + type_)()
        getattr(self, 'display_stats_' + type_)()

    # endregion

    # region Methods compute_

    @BaseClass.Verbose("Computing the database' article...")
    @BaseClass.Attribute('articles')
    def compute_articles(self):
        """ Computes and initializes the articles in the database. """

        articles = {}
        root = self.project_root + 'databases/nyt_jingyun/'

        for data_path in self.paths():
            id_ = data_path.split('/')[-1].split('.')[0]
            year = data_path.split('/')[-4]

            content_path = root + 'content_annotated/' + str(year) + 'content_annotated/' + id_ + '.txt.xml'
            summary_path = root + 'summary_annotated/' + str(year) + 'summary_annotated/' + id_ + '.txt.xml'

            articles[id_] = Article(data_path=data_path, content_path=content_path, summary_path=summary_path)

        self.articles = articles

    @BaseClass.Verbose("Computing the articles' metadata...")
    def compute_metadata(self):
        """ Computes the metadata of the articles. """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            self.articles[id_].compute_metadata()

    @BaseClass.Verbose("Computing the database' entities...")
    @BaseClass.Attribute('entities')
    def compute_entities(self):
        """ Compute the entities of the database. """

        self.entities = dict()

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')

            entities = self.articles[id_].get_entities()

            for entity in entities:
                if entity.name in self.entities:
                    self.entities[entity.name].update_info(entity)
                else:
                    self.entities[entity.name] = entity

            self.articles[id_].entities = {self.entities[name] for name in {entity.name for entity in entities}}

    @BaseClass.Verbose("Computing the entity tuples...")
    @BaseClass.Attribute('tuples')
    def compute_tuples(self):
        """ Compute the Tuples of the database as a sorted list of Tuples (by number of articles). """

        ids = defaultdict(set)

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')

            entities = defaultdict(set)
            for entity in self.articles[id_].entities:
                entities[entity.type_].add(entity.name)

            for type_ in entities:
                for tuple_ in self.subtuples(entities[type_]):
                    ids[tuple_].add(id_)

        ranking = sorted(ids, key=lambda k: len(ids[k]), reverse=True)

        self.tuples = [Tuple(id_=str(rank + 1),
                             entities=tuple([self.entities[name] for name in tuple_]),
                             article_ids=ids[tuple_])
                       for rank, tuple_ in enumerate(ranking)]

    @BaseClass.Verbose("Computing the articles' annotations...")
    def compute_annotations(self):
        """ Computes the annotations of the articles. """

        count, size = 0, len(self.articles)
        for id_ in self.articles:
            count = self.progression(count, self.modulo_articles, size, 'article')
            self.articles[id_].compute_annotations()

    @BaseClass.Verbose("Computing the contexts...")
    def compute_contexts(self):
        """ Compute the contexts of the articles for each Tuple. """

        count, size = 0, len(self.tuples)
        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')
            query_ids = set()

            for article_id_ in tuple_.article_ids:
                self.articles[article_id_].compute_contexts(tuple_=tuple_)

                query_ids.update({tuple_.id_ + '_' + article_id_ + '_' + context_id_
                                  for context_id_ in self.articles[article_id_].contexts[tuple_.get_name()]})

            tuple_.query_ids = query_ids

    @BaseClass.Verbose("Computing the Wikipedia information...")
    def compute_wikipedia(self, load):
        """
        Compute the wikipedia information about the entities from self.tuples.

        Args:
            load: bool, if True, load an existing file.
        """

        wikipedia = {'found': dict(), 'not_found': set()} if not load else self.wikipedia

        count, size = 0, len(self.entities)
        for name, entity in self.entities.items():
            count = self.progression(count, self.modulo_entities, size, 'entity')

            if not load:
                wiki = entity.get_wiki()

                if wiki.info is not None:
                    wikipedia['found'][name] = wiki
                else:
                    wikipedia['not_found'].add(name)

            else:
                if name in wikipedia['found']:
                    wiki = wikipedia['found'][name]
                elif name in wikipedia['not_found']:
                    wiki = Wikipedia(None, None)
                else:
                    print("The entity ({}) is not in the loaded wikipedia file.".format(str(entity)))
                    wiki = entity.get_wiki()

            entity.wiki = wiki

        print("Number of entities found/not found: {}/{}".format(len(wikipedia['found']), len(wikipedia['not_found'])))

        self.wikipedia = wikipedia

    @BaseClass.Verbose("Computing the Queries...")
    @BaseClass.Attribute('queries')
    def compute_queries(self):
        """ Compute the Queries of the database. """

        queries = dict()

        count, size = 0, len(self.tuples)
        for tuple_ in self.tuples:
            count = self.progression(count, self.modulo_tuples, size, 'tuple')

            for article_id_ in tuple_.article_ids:
                article_contexts = self.articles[article_id_].contexts[tuple_.get_name()]

                for context_id_ in article_contexts:
                    query_id_ = tuple_.id_ + '_' + article_id_ + '_' + context_id_
                    queries[query_id_] = Query(id_=query_id_,
                                               tuple_=tuple_,
                                               article=self.articles[article_id_],
                                               context=article_contexts[context_id_])

        self.queries = queries

    # endregion

    # region Statistics methods
    # TODO: change all the functions

    def compute_stats_tuples(self):
        """ Compute the entities tuples statistics of the database. """

        self.stats = self.stats or dict()

        self.stats['tuples_lengths'] = self.stat_tuples_lengths()
        self.stats['tuples_frequencies'] = self.stat_tuples_frequencies()
        self.stats['tuples_thresholds'] = self.stat_tuples_thresholds()

    def compute_stats_wikipedia(self):
        """ Compute the wikipedia statistics of the database. """

        self.stats = self.stats or dict()

        self.stats['wikipedia_length'] = self.stat_wikipedia_length('wikipedia')
        self.stats['notwikipedia_length'] = self.stat_wikipedia_length('not_wikipedia')
        self.stats['ambiguous_length'] = self.stat_wikipedia_length('ambiguous')
        self.stats['wikipedia_frequencies'] = self.stat_wikipedia_frequencies('wikipedia')
        self.stats['notwikipedia_frequencies'] = self.stat_wikipedia_frequencies('not_wikipedia')

    def compute_stats_contexts(self):
        """ Compute the contexts statistics of the database. """

        self.stats = self.stats or dict()

        self.stats['contexts'] = self.stat_contexts()

    def stat_tuples_lengths(self):
        """
        Compute the histogram of the lengths of the tuples as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the lengths of the entities tuples, starting from 0.
        """

        data = [len(tuple_.entities) for tuple_ in self.tuples]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_tuples_frequencies(self):
        """
        Compute the histogram of the frequencies of the tuples as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the frequencies of the entities tuples, starting from 0.
        """

        data = [len(tuple_.article_ids) for tuple_ in self.tuples]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_tuples_thresholds(self):
        """
        Compute the histogram of the size of the database corresponding to each threshold over the entities tuples
        frequency, starting from 0 (no threshold), as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the number of articles for each threshold.
        """

        m = max([len(tuple_.article_ids) for tuple_ in self.tuples])

        threshold_ids = [set() for _ in range(m + 1)]
        threshold_ids[0].update(set(self.articles.keys()))

        for tuple_ in self.tuples:
            for threshold in range(1, len(tuple_.article_ids) + 1):
                threshold_ids[threshold].update(tuple_.article_ids)

        data = [i for i in range(m + 1) for _ in threshold_ids[i]]
        bins = m + 1
        range_ = (0, m + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_wikipedia_length(self, file):
        """
        Compute the number of entries in the file.

        Args:
            file: str, file to analyse, must be 'wikipedia', 'not_wikipedia' or 'ambiguous'.

        Returns:
            int, number of entries in the file.
        """

        file = getattr(self, file)

        if file is not None:
            return len(file)
        else:
            return 0

    def stat_wikipedia_frequencies(self, file):
        """
        Compute the histogram of the frequencies of the tuples where appear the entities from file as a numpy.histogram.

        Args:
            file: str, file to analyse, must be 'wikipedia', 'not_wikipedia' or 'ambiguous'.

        Returns:
            numpy.histogram, histogram of the frequencies of the entities tuples, starting from 0.
        """

        file = getattr(self, file)

        data = [len(tuple_.article_ids) for tuple_ in self.tuples for entity in tuple_.entities if entity in file]
        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def stat_contexts(self):
        """
        Compute the histogram of the number of contexts as a numpy.histogram.

        Returns:
            numpy.histogram, histogram of the number of contexts.
        """

        data = []
        for tuple_ in self.tuples:
            length = 0
            for id_ in tuple_.article_ids:
                length += len(self.articles[id_].contexts[tuple_.entities])

            data.append(length)

        bins = max(data) + 1
        range_ = (0, max(data) + 1)

        return histogram(data, bins=bins, range=range_)

    def display_stats_tuples(self):
        """ Display the entities tuples statistics of the database. """

        print("\nTotal number of tuples: {}".format(len(self.tuples)))
        print("\n10 most frequent tuples:")
        for tuple_ in self.tuples[:10]:
            print("{} (in {} articles)".format(self.to_string(tuple_.entities), len(tuple_.article_ids)))
        print()

        self.plot_hist(fig=1, data=self.stats['tuples_lengths'], xlabel='lengths', log=True,
                       title='Lengths of the tuples of entities')

        self.plot_hist(fig=2, data=self.stats['tuples_frequencies'], xlabel='frequencies', log=True,
                       title='Frequencies of the tuples of entities')

        self.plot_hist(fig=3, data=self.stats['tuples_thresholds'], xlabel='thresholds', log=True,
                       title='Number of articles for each threshold on the frequency')

    def display_stats_contexts(self):
        """ Display the contexts statistics of the database. """

        self.plot_hist(fig=6, data=self.stats['contexts'], xlabel='number of contexts', log=True,
                       title="Number of contexts found for each tuple")

    def display_stats_wikipedia(self):
        """ Display the wikipedia statistics of the database. """

        print("\nTotal number of wikipedia: {}/not_wikipedia: {}/ambiguous: {}"
              .format(self.stats['wikipedia_length'],
                      self.stats['notwikipedia_length'],
                      self.stats['ambiguous_length']))

        print("\nWikipedia info of 10 most frequent tuples:\n")
        for tuple_ in self.tuples[:10]:
            print(self.to_string(self.get_info(tuple_.entities)) + '\n')

        print("\nEntities not found in wikipedia:")
        for entity in self.not_wikipedia:
            print(entity + ' (' + self.not_wikipedia[entity] + ')')

        print("\nAmbiguous cases:")
        for entity in self.ambiguous:
            print(self.to_string(self.ambiguous[entity]) + ' (' + entity + ')')
        print()

        self.plot_hist(fig=4, data=self.stats['wikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities found in wikipedia')

        self.plot_hist(fig=5, data=self.stats['notwikipedia_frequencies'], xlabel='frequencies', log=True,
                       title='Tuple frequency of the entities not found in wikipedia')

    # endregion

    # region Cleaning methods

    @BaseClass.Verbose("Cleaning the database's articles...")
    @BaseClass.Attribute('articles')
    def clean_articles(self, criterion=None, to_del=None, to_keep=None):
        """
        Removes from the database the articles which meet the Article's criterion or whose ids are in to_del or are not
        in to_keep.

        Args:
            criterion: function, criterion that an article must meet to be removed.
            to_del: set, ids of the articles that must be removed.
            to_keep: set, ids of the articles that must be kept.
        """

        if criterion is not None and to_del is None and to_keep is None:
            print("Criterion: {}".format([line for line in criterion.__doc__.splitlines() if line][0][8:]))
            to_del = set()

            for id_ in self.articles:
                if criterion(self.articles[id_]):
                    to_del.add(id_)

        elif criterion is None and to_del is not None and to_keep is None:
            print("Criterion: remove the designated articles.")

        elif criterion is None and to_del is None and to_keep is not None:
            print("Criterion: keep only the designated articles.")
            to_del = set()

            for id_ in self.articles:
                if id_ not in to_keep:
                    to_del.add(id_)

        else:
            raise Exception("Either a criterion or to_del or to_keep must be specified.")

        for id_ in to_del:
            del self.articles[id_]

    @BaseClass.Verbose("Cleaning the database's tuples...")
    @BaseClass.Attribute('tuples')
    def clean_tuples(self, to_del=None, to_keep=None):
        """
        Removes from the database the tuples whose names are in to_del or are not in to_keep.

        Args:
            to_del: set, names of the tuples that must be removed.
            to_keep: set, names of the tuples that must be kept.
        """

        if to_del is not None and to_keep is None:
            print("Criterion: remove the designated tuples.")
            tuples = self.tuples
            self.tuples = []

            for tuple_ in tuples:
                if tuple_.get_name() not in to_del:
                    self.tuples.append(tuple_)

        elif to_del is None and to_keep is not None:
            print("Criterion: keep only the designated tuples.")
            tuples = self.tuples
            self.tuples = []

            for tuple_ in tuples:
                if tuple_.get_name() in to_keep:
                    self.tuples.append(tuple_)

        else:
            raise Exception("Either to_del or to_keep must be specified.")

    @BaseClass.Verbose("Cleaning the database's entities...")
    @BaseClass.Attribute('entities')
    def clean_entities(self, to_del=None, to_keep=None):
        """
        Removes from the database the entities whose names are in to_del or are not in to_keep.

        Args:
            to_del: set, names of the entities that must be removed.
            to_keep: set, names of the entities that must be kept.
        """

        if to_del is not None and to_keep is None:
            print("Criterion: remove the designated entities.")

        elif to_del is None and to_keep is not None:
            print("Criterion: keep only the designated entities.")
            to_del = set()

            for name in self.entities:
                if name not in to_keep:
                    to_del.add(name)

        else:
            raise Exception("Either to_del or to_keep must be specified.")

        for name in to_del:
            del self.entities[name]

    @BaseClass.Verbose("Filtering the articles, tuples and entities...")
    def filter(self, min_articles=None, min_queries=None):
        """
        Filter out the articles that doesn't respect the specified threshold on the minimum number of articles or the
        minimum number of queries.

        Args:
            min_articles: int, minimum number of articles an entities' tuple must be in.
            min_queries: int, minimum number of Queries an entities' tuple must have.
        """

        to_keep_articles, to_keep_tuples, to_keep_entities = set(), set(), set()

        if min_articles is not None and min_queries is None:
            print("Minimum number of articles: {}".format(min_articles))
            threshold = min_articles
            attribute = 'article_ids'
        elif min_articles is None and min_queries is not None:
            print("Minimum number of queries: {}".format(min_queries))
            threshold = min_queries
            attribute = 'query_ids'
        else:
            raise Exception("Either min_articles or min_queries must be specified.")

        for tuple_ in self.tuples:
            if len(getattr(tuple_, attribute)) >= threshold:
                to_keep_tuples.add(tuple_.get_name())
                to_keep_articles.update(tuple_.article_ids)
                to_keep_entities.update([entity.name for entity in tuple_.entities])

        self.clean_tuples(to_keep=to_keep_tuples)
        self.clean_articles(to_keep=to_keep_articles)
        self.clean_entities(to_keep=to_keep_entities)

        self.min_articles = min_articles if min_articles is not None else self.min_articles
        self.min_queries = min_queries if min_queries is not None else self.min_queries

    # endregion

    # region File methods

    def prefix_suffix(self):
        """
        Returns the standard beginning and ending for a file path.

        Returns:
            prefix: str, beginning of the name of the file (until the name of the folder).
            suffix: str, ending of the name of the file (after the basic name of the file).
        """

        year = str(self.years[0]) if len(self.years) == 1 else str(self.years[0]) + '-' + str(self.years[-1])[2:4]
        prefix, suffix = self.project_root + 'results/' + year + '/', ''

        if self.max_size is None:
            suffix += '_sizemax'
        elif self.max_size >= 1000:
            suffix += '_size' + str(self.max_size // 1000) + 'k'
        else:
            suffix += '_size' + str(self.max_size)

        if self.min_articles is not None:
            suffix += '_articles' + str(self.min_articles)
        if self.min_queries is not None:
            suffix += '_queries' + str(self.min_queries)

        return prefix, suffix

    def save_pkl(self, attribute_name=None, obj=None, file_name=None, folder_name='queries'):
        """
        Save an attribute (designated by its name) or an object into a file using pickle.

        Args:
            attribute_name: str, name of the attribute to save; if None, save an object instead.
            obj: unk, object saved if no attribute name is provided.
            file_name: str, name of the file; if None, save an attribute with the standard name.
            folder_name: str, name of the folder to save in.
        """

        if attribute_name is not None:
            obj = getattr(self, attribute_name)
        if obj is None:
            raise Exception("Nothing to save, object is None.")

        prefix, suffix = self.prefix_suffix()
        if file_name is not None:
            file_name = prefix + folder_name + '/' + file_name + suffix + '.pkl'
        else:
            if attribute_name is not None:
                file_name = prefix + folder_name + '/' + attribute_name + suffix + '.pkl'
            else:
                raise Exception("Missing file name to save the object.")

        try:
            with open(file_name, 'wb') as f:
                dump(obj=obj, file=f, protocol=-1)

            if attribute_name is not None:
                print("Attribute {} saved at {}.".format(attribute_name, file_name))
            else:
                print("Object saved at {}.".format(file_name))

        except PicklingError:
            print("Could not save (PicklingError).")

    def load_pkl(self, attribute_name=None, file_name=None, folder_name='queries'):
        """
        Load an attribute (designated by its name) or an object from a file using pickle.

        Args:
            attribute_name: str, name of the attribute to load; if None, returns the object.
            file_name: str, name of the file to load; if None, load the file with the corresponding standard name.
            folder_name: str, name of the folder to load from.
        """

        prefix, suffix = self.prefix_suffix()
        if file_name is not None:
            file_name = prefix + folder_name + '/' + file_name + suffix + '.pkl'
        else:
            if attribute_name is not None:
                file_name = prefix + folder_name + '/' + attribute_name + suffix + '.pkl'
            else:
                raise Exception("Missing file name to load the object.")

        with open(file_name, 'rb') as f:
            obj = load(f)

        if attribute_name is not None:
            print("Attribute {} loaded from {}.".format(attribute_name, file_name))
            setattr(self, attribute_name, obj)
        else:
            print("Object loaded from {}".format(file_name))
            return obj

    def save_csv(self, attribute_name=None, folder_name='queries', limit=None):
        """
        Save a dictionary attribute to a .csv using pandas DataFrame.

        Args:
            attribute_name: str, name of the attribute to save.
            folder_name: str, name of the folder to save in.
            limit: int, maximum number of data to save; if None, save all of them.
        """

        obj = getattr(self, attribute_name)
        ids = list(obj.keys())

        if limit is not None:
            seed(seed=42)
            shuffle(ids)
            ids = ids[:limit]

        data = [obj[id_].to_dict() for id_ in ids]
        df = DataFrame.from_records(data=data)

        prefix, suffix = self.prefix_suffix()
        file_name = attribute_name if limit is None else attribute_name + '_short'
        file_name = prefix + folder_name + '/' + file_name + suffix + '.csv'

        df.to_csv(file_name, index_label='idx')

        print("Attribute {} saved at {}".format(attribute_name, file_name))

    # endregion

    # region Other methods

    @staticmethod
    def progression(count, modulo, size, text):
        """
        Prints progression's updates and update the count.

        Args:
            count: int, current count.
            modulo: int, how often to print updates.
            size: int, size of the element to count.
            text: str, what to print at the beginning of the updates.

        Returns:
            int, incremented count of articles.
        """

        count += 1

        if count % modulo == 0:
            print("  " + text + " {}/{}...".format(count, size))

        return count

    def paths(self):
        """
        Compute the paths of the data files of the database.

        Returns:
            list, sorted file paths of the data of the articles.
        """

        patterns = [self.project_root + 'databases/nyt_jingyun/data/' + str(year) + '/*/*/*.xml' for year in self.years]

        paths = []
        for pattern in patterns:
            paths.extend(glob(pattern))

        paths.sort()

        return paths[:self.max_size] if self.max_size is not None else paths

    @staticmethod
    def subtuples(l):
        """
        Compute all the possible sorted subtuples of len > 1 from a list.

        Args:
            l: list, original list.

        Returns:
            set, all the possible subtuples of len > 1 of l.
        """

        if len(l) < 2:
            return set()

        elif len(l) == 2 or len(l) > 10:
            return {tuple(sorted(l))}

        else:
            res = {tuple(sorted(l))}
            for x in l:
                res = res.union(Database.subtuples([y for y in l if y != x]))

            return res

    @staticmethod
    def plot_hist(fig, data, title, xlabel, log=False):
        """
        Plot the data as a histogram using matplotlib.pyplot. Print the data as well.

        Args:
            fig: int, index of the figure.
            data: numpy.histogram, histogram of the data.
            title: str, title of the figure.
            xlabel: str, label of the x-axis.
            log: bool, whether to use a logarithmic scale or not.
        """

        plt.figure(num=fig, figsize=(12, 4))

        counts, bins = data
        plt.hist(bins[:-1], bins, weights=counts, align='left', rwidth=.8, log=log)
        plt.title(title)
        plt.xlabel(xlabel)

    # endregion


def main():
    max_size = 1000
    min_articles = 1
    min_queries = 1

    database = Database(project_root='../', max_size=max_size, min_articles=min_articles, min_queries=min_queries)

    database.preprocess_database()
    database.process_articles()

    database.process_wikipedia(load=True)
    database.process_queries(load=True)
    return


if __name__ == '__main__':
    main()
