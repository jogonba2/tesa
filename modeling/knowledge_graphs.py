import pwb
import pywikibot
from breaking_cycles import remove_cycle_edges_by_hierarchy as rce
from pywikibot import config, pagegenerators
from pywikibot.tools import (
    deprecated_args, deprecated, ModuleDeprecationWrapper, open_archive,
)
from typing import Set
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import has_path
import numpy as np
import functools
import itertools
import networkx as nx
import pickle as pkl
import os
import re
import mwparserfromhell


class KnowledgeGraph:

    def __init__(self, filename):
        """
        Builds an empty graph as default.
        Parameters:
            * filename: str, path to save the category graph.
        """
        self.filename = filename
        self.graph = nx.DiGraph()

    def set_graph(self, graph):
        self.graph = graph

    def save(self) -> None:
        with open(self.filename, "wb") as fw:
            pkl.dump(self.graph, fw)

    def load(self) -> None:
        with open(self.filename, "rb") as fr:
            self.graph = pkl.load(fr)

    def traverse(self):
        pass

    def build_graph(self, entities):
        self.graph.add_nodes_from(entities)
        self.save()

    def generate_dataset(self):
        """
        Generates the dataset of sentences for learning Knowledge Graph Embeddings.
        """
        pass

class InfoboxGraph(KnowledgeGraph):

    def __init__(self, filename="infobox_graph.pkl", cachename="infobox_dump.pkl"):
        super(InfoboxGraph, self).__init__(filename)
        self.parsers = {str(f)[6:]: getattr(InfoboxParsers, f)
                        for f in dir(InfoboxParsers) if f.startswith("parse_")}
        pywikibot.Site().login()
        self.site = pywikibot.Site()
        self.cachename = cachename
        self.get_resources_regex = re.compile("(\[\[.*?\]\])")
        self.cache = self.load_cache()


    def preprocess_predicate(self, pred):
        """
        Method for preprocessing a predicate of a triple.
            1. Remove numbers for multiple predicates (office, office2, ..., office_n)
            2. Replace _
            3. Lowering
            4. Remove :en:
            4. TODO
        Args:
            * pred: str, predicate of a triple
        """

        pred = "".join(filter(lambda c: not c.isdigit(), pred))
        pred = pred.replace("_", " ")
        pred = pred.lower()
        pred = pred.replace(":en:", "")
        return pred

    def preprocess_object(self, obj):
        """
        Method for preprocessing an object of a triple.
            1. Forall objects that represent Wikipedia pages ([[.*]]):
                1.1 Get the resource name (first element before |)
                1.2 Remove #
                1.3 Remove [ and ]
            2. Remove "lists of" and "list of"
            3. Remove "by ..." suffix
            4. TODO
         Args:
             * pred: str, predicate of a triple
         """
        objs = self.get_resources_regex.findall(obj)
        for i, obj in enumerate(objs):
            obj = obj.split("|")[0].split("#")[0]
            obj = obj.replace("[", "").replace("]", "")
            obj = obj.replace("List of", "").replace("Lists of", "")
            obj_words = obj.split()
            if "by" in obj_words:
                obj = " ".join(obj_words[:obj_words.index("by")])
            objs[i] = obj
        return objs

    def is_valid_predicate(self, pred):
        """
        Check valid predicates:
            1. TODO
        """
        if not pred:
            return False
        return True

    def is_valid_object(self, obj):
        """
        Check valid objects:
            1. If not starts by File:
            2. TODO
        """
        if obj.startswith("File:"):
            return False

        return True

    def remove_independent_nodes(self):
        """
        Remove nodes u | d_i(u) + d_o(u) = 1
        """
        independent_nodes = [node for node in self.graph.nodes if self.graph.degree[node] == 1]
        for node in independent_nodes:
            self.graph.remove_node(node)

    def get_entities(self):
        return [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

    def parse_infobox(self, infobox):
        """
        Parse a given infobox. Applies the InfoboxParsers for filtering and extracting
        additional information. Also makes a general preprocess of predicates and objects.
        Args:
            * infobox: dict, infobox dictionary
        Returns:
            * pred_obj_pairs: list, list of (predicate, object) pairs extracted from the infobox.
        """

        pred_obj_pairs = []
        for pred, obj in list(infobox.items()):
            pred = self.preprocess_predicate(pred)
            if self.is_valid_predicate(pred):
                # region InfoboxParser
                if pred in self.parsers:
                    filtered_info = self.parsers[pred](obj)
                    for (p, o) in filtered_info:
                        if o is not None and self.is_valid_object(o) and self.is_valid_predicate(p):
                            pred_obj_pairs.append((p, o))
                # endregion
                # region ResourceParser
                if obj is not None and type(obj)==str and "[[" in obj:
                    objs = self.preprocess_object(obj)
                    for obj in objs:
                        if self.is_valid_object(obj):
                            pred_obj_pairs.append((pred, obj))
                # endregion

        return pred_obj_pairs

    def traverse(self, entity, depth, max_depth):
        if depth < max_depth:
            if entity and entity in self.cache:
                pred_obj_pairs = self.cache[entity]
                self.graph.add_edges_from([(entity, obj, {"label": pred})
                                           for (pred, obj) in pred_obj_pairs])
                for (_, obj) in pred_obj_pairs:
                    return self.traverse(obj, depth + 1, max_depth)

            else:
                if not entity:
                    return

                entity_page = list(self.site.search(entity, total=1, where="title"))

                if entity_page:
                    entity_page = entity_page[0]

                    try:
                        templates = mwparserfromhell.parse(entity_page.get()).filter_templates()
                        infobox = self.get_infobox(templates)
                    except:
                        return

                    if infobox is not None:
                        pred_obj_pairs = self.parse_infobox(infobox)
                        self.cache[entity] = pred_obj_pairs
                        self.graph.add_edges_from([(entity, obj, {"label": pred})
                                                   for (pred, obj) in pred_obj_pairs])
                        for (_, obj) in pred_obj_pairs:
                            return self.traverse(obj, depth + 1, max_depth)
        return

    def get_infobox(self, templates):
        infobox = {}
        for template in templates:
            if "infobox" in template.lower():
                break

        for param in template.params:
            infobox[str(param.name).strip()] = str(param.value).strip()

        return infobox

    def save_cache(self) -> None:
        with open(self.cachename, "wb") as fw:
            pkl.dump(self.cache, fw)

    def load_cache(self) -> None:
        if os.path.isfile(self.cachename):
            with open(self.cachename, "rb") as fr:
                return pkl.load(fr)
        else:
            return {}

    def build_graph(self, entities, max_depth):
        """
        Build and save the infobox graph extracted from all the entities of the TESA dataset.
        """
        for entity in tqdm(entities):
            if not entity:
                continue
            entity_page = list(self.site.search(entity, total=1, where="title"))
            if entity_page:
                entity_page = entity_page[0]
                try:
                    templates = mwparserfromhell.parse(entity_page.get()).filter_templates()
                    infobox = self.get_infobox(templates)
                except:
                    continue

                if infobox is not None:
                    pred_obj_pairs = self.parse_infobox(infobox)
                    self.cache[entity] = pred_obj_pairs
                    self.graph.add_edges_from([(entity, obj, {"label": pred}) for (pred, obj) in pred_obj_pairs])
                    for (_, obj) in pred_obj_pairs:
                        self.traverse(obj, 1, max_depth)

        self.save()
        self.save_cache()

class FullGraph(KnowledgeGraph):

    def __init__(self, filename="full_graph.pkl"):
        super(FullGraph, self).__init__(filename)

    def build_graph(self, infobox_graph, category_graph):
        self.graph = nx.compose(infobox_graph, category_graph)
        self.save()

    def generate_dataset(self):
        pass


class CategoryGraph(KnowledgeGraph):

    """
    Robot for creating the category graph for the TESA task:

    Category graph: G(V,E) | V are categories and entites and exists (u, v) if u is a category of v.
                    Edges are labeled by "is a" (although "is a" is not guaranteed)
                    for higher levels.

    """

    def __init__(self, cat_db, filename="category_graph.pkl"):
        super(CategoryGraph, self).__init__(filename)
        self.cat_db = cat_db
        pywikibot.Site().login()
        self.site = pywikibot.Site()
        self.dag = None
        self.category_roots = ['academic disciplines', 'business', 'concepts', 'crime',
                               'culture', 'economy', 'education', 'energy', 'engineering',
                               'entertainment', 'events', 'food and drink', 'geography',
                               'government', 'health', 'history', 'human behavior', 'humanities',
                               'industry', 'knowledge', 'language', 'law', 'life', 'mass media',
                               'mathematics', 'military', 'mind', 'music', 'nature', 'objects',
                               'organizations', 'people', 'philosophy', 'policy', 'politics', 'religion',
                               'science and technology', 'society', 'sports', 'universe', 'world']

    def clean_category_title(self, raw_title):
        for k in [":Category:", "]", "[", "en:"]:
            raw_title = raw_title.replace(k, "")
        return raw_title.split("|")[0]

    def is_valid_category(self, title):
        l_title = title.lower()
        for k in ["wikipedia", "hidden", "category", "categories",
                  "errors", "pages", "births", "catautotoc", "dates",
                  "articles", "alumni", "descent", "sources", "properties",
                  "template", "maint", "maintenance", "wikidata", "npov", "cs1",
                  "century", "calendar", "chronology", "lists of", "list of"]:

            if k in l_title:
                return False

        for k in self.category_roots:
            if k == l_title:
                return False

        return True

    def add_category_edge(self, graph, sub_cat_title, super_cat_title, label):
        """
        Add edges to the graph taking into account some details:
        * (subcategory, supercategory by ...) -> add (subcategory, supercategory) edge.
        * avoid loops
        * ...
        """
        if "by" in super_cat_title:
            super_cat_title = super_cat_title[:super_cat_title.index("by")].strip()
        if "by" in sub_cat_title:
            sub_cat_title = sub_cat_title[:sub_cat_title.index("by")].strip()
        if super_cat_title != sub_cat_title:
            graph.add_edge(sub_cat_title, super_cat_title, label=label)

    def traverse(self, graph, category, depth, max_depth):
        if depth < max_depth:
            category_title = self.clean_category_title(category.title(as_link=True,
                                                                      textlink=True,
                                                                      with_ns=False))
            if self.is_valid_category(category_title):
                super_cats = self.cat_db.getSupercats(category)
                for super_cat in super_cats:
                    super_cat_title = self.clean_category_title(super_cat.title(as_link=True,
                                                                                textlink=True,
                                                                                with_ns=False))
                    if self.is_valid_category(super_cat_title):
                        self.add_category_edge(graph, category_title, super_cat_title, label="is a")
                        self.traverse(graph, super_cat, depth + 1, max_depth)
        return

    def get_entities(self):
        return [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

    def build_graph(self, entities, max_depth) -> None:
        """
        Build and save the category graph extracted from all the entities of the TESA dataset.
        Build and save the CategoryDatabase object as category buffer for new entities.

        Args:
            * entities: list, list of entitie's names
            * max_depth: int, max depth for the traversals, starting from each entity.
        """
        for entity in tqdm(entities):
            entity_page = list(self.site.search(entity, total=1, where="title"))
            if entity_page:
                entity_page = entity_page[0]
                categories = list(entity_page.categories())
                for cat in categories:
                    # Add edge between entity and its direct categories
                    cat_title = self.clean_category_title(cat.title(as_link=True,
                                                                    textlink=True,
                                                                    with_ns=False))
                    if self.is_valid_category(cat_title):
                        self.add_category_edge(self.graph, entity, cat_title, label="is a")
                    # Add edges between categories and subcategories
                    self.traverse(self.graph, cat, 1, max_depth)
        self.save()

    def build_dag(self, fname_dag) -> None:
        """
        Builds and saves the DAG computed from the category graph.

        Args:
            fname_dag: str, filename for saving the dag
        """

        self.dag = nx.DiGraph()
        graph = self.graph.reverse()
        indices = {node: i for i, node in enumerate(graph.nodes)}
        relabeled_graph = nx.relabel_nodes(graph, indices)

        self.dag = rce.build_dag(relabeled_graph, "trueskill", fname_dag)

        self.dag = nx.relabel_nodes(self.dag, {v: k for k, v in indices.items()})

        with open(fname_dag, "wb") as fw:
            pkl.dump(self.dag, fw)

class InfoboxParsers:

    """
    parse_key methods are used by the InfoboxGraph class for extracting additional filtered pairs of (p, o),
              the key substring indicates the use of the parser when predicate=key
    """

    # region parse_ #
    @staticmethod
    def parse_coordinates(obj):
        """
        The parser fails with some coordinate formats, but, if clean
        different information can be extracted:
            1. Coordinates
            2. Region (ES, FR, GB, ..)
        Args:
            * obj: str, object of a triple.

        Returns:
            * filtered_info: list, new information extracted from the object.
        """

        obj = obj.replace("}", "").replace("{", "").split("|")[1:]

        # region Coordinates #
        coordinates = None
        coord_delim = [obj.index(k) for k in ["E", "W"] if k in obj]
        coord_delim = coord_delim[0] if coord_delim else None

        if coord_delim is not None:
            coordinates = " ".join(obj[:coord_delim + 1])  # Specify degrees

        # endregion Coordinates #

        # region CoordinatesRegion #
        coord_region = None
        for k in obj:
            if "region" in k:
                coord_region = k.split(":")[1]
                break
        # endregion CoordinatesRegion #

        return [("coordinates", coordinates),
                ("region", coord_region)]

    # endregion


class CategoryDatabase:

    """Temporary database saving pages and subcategories for each category.

       This prevents loading the category pages over and over again.
    """

    def __init__(self, rebuild=False, filename='category.dump.bz2') -> None:
        """Initializer."""
        self.filename = filename
        if rebuild:
            self.rebuild()

    @property
    def is_loaded(self) -> bool:
        """Return whether the contents have been loaded."""
        return hasattr(self, 'catContentDB') and hasattr(self, 'superclassDB')

    def _load(self) -> None:
        if not self.is_loaded:
            try:
                if config.verbose_output:
                    pywikibot.output('Reading dump from '
                                     + config.shortpath(self.filename))
                with open_archive(self.filename, 'rb') as f:
                    databases = pkl.load(f)
                # keys are categories, values are 2-tuples with lists as
                # entries.
                self.catContentDB = databases['catContentDB']
                # like the above, but for supercategories
                self.superclassDB = databases['superclassDB']
                del databases
            except Exception as e:
                # If something goes wrong, just rebuild the database
                self.rebuild()

    def rebuild(self) -> None:
        """Rebuild the dabatase."""
        self.catContentDB = {}
        self.superclassDB = {}

    def getSubcats(self, supercat) -> Set[pywikibot.Category]:
        """Return the list of subcategories for a given supercategory.

        Saves this list in a temporary database so that it won't be loaded
        from the server next time it's required.
        """
        self._load()
        # if we already know which subcategories exist here
        if supercat in self.catContentDB:
            return self.catContentDB[supercat][0]
        else:
            subcatset = set(supercat.subcategories())
            articleset = set(supercat.articles())
            # add to dictionary
            self.catContentDB[supercat] = (subcatset, articleset)
            return subcatset

    def getArticles(self, cat) -> Set[pywikibot.Page]:
        """Return the list of pages for a given category.

        Saves this list in a temporary database so that it won't be loaded
        from the server next time it's required.
        """
        self._load()
        # if we already know which articles exist here.
        if cat in self.catContentDB:
            return self.catContentDB[cat][1]
        else:
            subcatset = set(cat.subcategories())
            articleset = set(cat.articles())
            # add to dictionary
            self.catContentDB[cat] = (subcatset, articleset)
            return articleset

    def getSupercats(self, subcat) -> Set[pywikibot.Category]:
        """Return the supercategory (or a set of) for a given subcategory."""
        self._load()
        # if we already know which subcategories exist here.
        if subcat in self.superclassDB:
            return self.superclassDB[subcat]
        else:
            supercatset = set(subcat.categories())
            # add to dictionary
            self.superclassDB[subcat] = supercatset
            return supercatset

    def dump(self, filename=None) -> None:
        """Save the dictionaries to disk if not empty.

        Pickle the contents of the dictionaries superclassDB and catContentDB
        if at least one is not empty. If both are empty, removes the file from
        the disk.

        If the filename is None, it'll use the filename determined in __init__.
        """
        if filename is None:
            filename = self.filename
        elif not os.path.isabs(filename):
            filename = config.datafilepath(filename)
        if self.is_loaded and (self.catContentDB or self.superclassDB):
            pywikibot.output('Dumping to {}, please wait...'
                             .format(config.shortpath(filename)))
            databases = {
                'catContentDB': self.catContentDB,
                'superclassDB': self.superclassDB
            }
            # store dump to disk in binary format
            with open_archive(filename, 'wb') as f:
                try:
                    pkl.dump(databases, f, protocol=config.pickle_protocol)
                except pkl.PicklingError:
                    pass
        else:
            try:
                os.remove(filename)
            except EnvironmentError:
                pass
            else:
                pywikibot.output('Database is empty. {} removed'
                                 .format(config.shortpath(filename)))

class GraphAlgorithms:

    @staticmethod
    # TODO remove duplicates and entities in TESA task
    def extreme_candidates(graph, nodes, n_candidates, tesa_entities, already_candidates):

        # If it's not a valid_query, sample random nodes from the graph
        if not GraphAlgorithms.is_valid_query(graph, nodes):
            diff = set([node.lower() for node in graph.nodes()])
            for set_ in [tesa_entities, already_candidates]:
                diff = diff.difference(set_)
            return set(np.random.choice(list(diff),
                                    size=n_candidates,
                                    replace=False))

        # Step 1: compute a maximum of n_candidates LCA #
        negative_candidates = set([lca.lower() for lca in
                                   GraphAlgorithms.lowest_common_ancestors(graph.reverse(copy=True),
                                                                          nodes, top_k=n_candidates)])

        # Step 2: if there aren't enough candidates, union of the direct neighbors of each entity
        if len(negative_candidates) < n_candidates:
            negative_candidates = negative_candidates.union(set([ind.lower() for ind in
                                                                  itertools.chain(*[list(graph.successors(node))
                                                                                  for node in nodes])]))

        # Step 3: delete candidates that are entities of TESA
        negative_candidates = negative_candidates.difference(tesa_entities)

        # Step 4: delete candidates that already appear in the original ranking
        negative_candidates = negative_candidates.difference(already_candidates)

        # Step 5: if there aren't enough candidates, pick randomly from the graph (sample from the difference
        # to avoid repetitions and tesa entities!)
        if len(negative_candidates) < n_candidates:
            diff = set([node.lower() for node in graph.nodes()])
            for set_ in [negative_candidates, tesa_entities, already_candidates]:
                diff = diff.difference(set_)
            negative_candidates = negative_candidates.union(np.random.choice(list(diff), replace=False,
                                                                            size=n_candidates - len(negative_candidates)
                                                                             ))

        return negative_candidates

    @staticmethod
    def negatives_from_neighborhood(graph, nodes, **kwargs):
        """
        $S^- = \bigcup\limits_{u\in T}\{v \ /\ (u, v) \in \textrm{E}\ \wedge \not\exists \ p(u', v)\ \forall u' \in T\}$
        """
        if not GraphAlgorithms.is_valid_query(graph, nodes):
            return []
        candidates = {node: list(graph.successors(node)) for node in nodes}
        negative_candidates = []
        for e1 in candidates:
            for successor in candidates[e1]:
                paths = 0
                for e2 in candidates:
                    if e2 != e1:
                        if has_path(graph, e2, successor):
                            paths += 1
                if paths < len(nodes) - 1:
                    negative_candidates.append(successor)

        return sorted(negative_candidates)

    @staticmethod
    def lowest_common_ancestors_graph(graph, nodes, top_k=6, **kwargs):
        """
        Wrapper for lowest common ancestors on the category graph
        """
        graph = graph.reverse()
        return GraphAlgorithms.lowest_common_ancestors(graph, nodes, top_k)

    @staticmethod
    def lowest_common_ancestors(graph, nodes, top_k=6, **kwargs):
        """
        It computes the "nearest common ancestors" in the sense that it sorts the common ancestors
        for minimizing the sum of distances to the source nodes (entities).
        Args:
            graph: networkx.DiGraph
            nodes: list, list of source nodes.
            top_k: number of common ancestors to extract
        """

        def filtered_distance(term, dist):
            if term in ["Living people", "People", "Places",
                        "Use American English", "Use British English", "EngvarB"] \
                    or any(list(map(lambda x: x in term.lower(), ["year of", "grey links",
                                                                  "accuracy disputes", "deaths",
                                                                  "births"]))):
                return float("inf")
            else:
                return dist

        if not GraphAlgorithms.is_valid_query(graph, nodes):
            return []

        common_ancestors = list(set.intersection(*[nx.ancestors(graph, node).union(set([node])) for
                                                   node in nodes]))

        if not common_ancestors:
            return []

        sum_of_path_lengths = np.zeros((len(common_ancestors)))
        for ii, c in enumerate(common_ancestors):
            sum_of_path_lengths[ii] = functools.reduce(lambda x, y: x + y,
                                                       [nx.shortest_path_length(graph, c, node)
                                                        for node in nodes])
            sum_of_path_lengths[ii] = filtered_distance(c, sum_of_path_lengths[ii])

        indices = np.argsort(sum_of_path_lengths)[:top_k]
        return sorted([common_ancestors[ii] for ii in indices])

    @staticmethod
    def sisp(graph, nodes, **kwargs):
        """
        Gets the Subgraph Induced by the Shortest Path (SISP)
        from each entity (node) to the lowest common ancestors
        """
        pass

    @staticmethod
    def random_walk(graph, nodes, **kwargs):
        pass

    @staticmethod
    def is_valid_query(graph, nodes, **kwargs):
        return all([graph.has_node(node) for node in nodes])

    @staticmethod
    def get_successors_hops(graph, node, hops=1, **kwargs):
        """
        Computes the successors until depth hops of the infobox graph.
        Args:
            node: str, entity to compute the hops
            graph: nx.DiGraph
            hops: int, maximum depth for compute the successors
        Returns:
            hops_map: dict, e.g. {entity1: {0: {...}, 1: {...}, ..., "all": {...}}, ...
                                  entityn:  {0: {...}, 1: {...}, ..., "all": {...}}}
        """
        hops_map = {0: set([node])}
        for i in range(1, hops + 1):
            hops_map[i] = set([])
            if node not in graph:
                continue
            for predecessor in hops_map[i - 1]:
                successors = set([successor for successor in graph.successors(predecessor)
                                  if successor and "WP:" not in successor])
                hops_map[i] = hops_map[i].union(successors)

        if node in graph:
            hops_map["all"] = set.union(*list(hops_map.values()))
        else:
            hops_map["all"] = hops_map[0]
        return hops_map

    @staticmethod
    def sort_intersections_by_depth(intersection, entities_hops, **kwargs):
        """
        Sorts the intersection among entities by the sum of distances to the entities.
        Args:
            intersection: set, shared nodes between the entities.
            entities_hops: dict, computed by get_successors_hops.
        Returns:
            sorted_intersections: list of intersections sorted by sum of distances.
        """
        depth_intersection = []
        for node in intersection:
            sum_depths = 0
            for entity in entities_hops:
                for depth in entities_hops[entity]:
                    if depth != "all" and node in entities_hops[entity][depth]:
                        sum_depths += depth
                        break
            depth_intersection.append((node, sum_depths))
        sorted_intersections = [k for k, _ in sorted(list(set(depth_intersection)), key=lambda x: x[1])]
        return sorted_intersections

    @staticmethod
    def value_intersection_infobox(graph, nodes, hops=1, k=999, **kwargs):
        """
        Given a tuple of entities
        1) Computes the successors of each entity until a maximum depth *hops*.
        2) Intersects all successors (common successors)
        3) Sorts the common successors by sum of distances to the entities and pick the $k$ lowest.

        Args:
            graph: nx.DiGraph.
            nodes: entity tuple from the TESA task.
            hops: int, maximum depth for computing the successors.
            k: int, for selecting the $k$ lowest.
        """

        node_hops = {node: GraphAlgorithms.get_successors_hops(graph, node, hops) for node in nodes}
        all_hops = [node_hops[node]["all"] for node in node_hops]
        intersection = set.intersection(*all_hops)
        intersection = GraphAlgorithms.sort_intersections_by_depth(intersection, node_hops)[:k]
        return intersection