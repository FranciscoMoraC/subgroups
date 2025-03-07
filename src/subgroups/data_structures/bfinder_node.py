# -*- coding: utf-8 -*-

# Contributors:

"""This file contains the implementation of 
"""


class BFinderNode(object):
    """
    This class represents a graph node of the BerryFinder algorithm.
    :param rank: rank of the node.
    """
    __slots__ = ["_selector_indexes","_descendant_nodes", "_rank" , "_max_descendant_rank", "_redundancies_checked"]

    def __init__(self, selector_indexes : set, rank : int):
        if type(selector_indexes) is not set:
            raise TypeError("selector_indexes must be a set.")
        if type(rank) is not int:
            raise TypeError("rank must be an int.")
        # Indexes of the selectors that the node contains in the list of all selectors.
        self._selector_indexes = selector_indexes
        # List of descendant nodes.
        self._descendant_nodes = []
        self._rank = rank
        self._max_descendant_rank = -1
        self._redundancies_checked = False

    def _get_descendant_nodes(self):
        """
        This method returns the list of descendant nodes.
        """
        return self._descendant_nodes
    
    def _get_selector_indexes(self):
        """
        This method returns the indexes of the selectors that the node contains.
        """
        return self._selector_indexes
    
    def _get_descendant_indexes(self):
        """
        This method returns a list of sets of indexes of the selectors that the descendant nodes contain.
        """
        return [descendant_node.selector_indexes for descendant_node in self._descendant_nodes]
    
    def _get_rank(self):
        """
        This method returns the rank of the node.
        """
        return self._rank
    
    def _get_redundancies_checked(self):
        """
        This method returns the value of the attribute redundancies_checked.
        """
        return self._redundancies_checked
    
    def _set_redundancies_checked(self, value : bool):
        """
        This method sets the value of the attribute redundancies_checked.
        :param value: value to be set.
        """
        if type(value) is not bool:
            raise TypeError("value must be a bool.")
        self._redundancies_checked = value

    def _get_max_descendant_rank(self):
        """
        This method returns the maximum rank of the descendant nodes.
        """
        return self._max_descendant_rank
    
    def _set_max_descendant_rank(self, value : int):
        """
        This method sets the maximum rank of the descendant nodes.
        :param value: value to be set.
        """
        if type(value) is not int:
            raise TypeError("value must be an int.")
        self._max_descendant_rank = value

    descendant_nodes = property(_get_descendant_nodes, None, None, "List of descendant nodes.")
    selector_indexes = property(_get_selector_indexes, None, None, "Set of indexes of the selectors that the node contains.")
    descendant_indexes = property(_get_descendant_indexes, None, None, "List of sets of indexes of the selectors that the descendant nodes contain.")
    rank = property(_get_rank, None, None, "Rank of the pattern represented by the node.")
    redundancies_checked = property(_get_redundancies_checked, _set_redundancies_checked, None, "Boolean attribute that indicates if the redundancies of the node have been checked.")
    max_descendant_rank = property(_get_max_descendant_rank, _set_max_descendant_rank, None, "Maximum rank of the descendant nodes.")

    def is_discarded(self):
        """
        This method returns a boolean attribute that indicates if the node has been discarded.
        """
        # The root node is always discarded
        if self.rank == -1:
            return True
        return self._max_descendant_rank >= self._rank

    def get_refining_descendants(self, set_of_selector_indexes : set):
        """
        This method returns the descendant nodes that refine a provided set of selectors.
        :param set_of_selector_indexes: set of indexes of the selectors that the descendant nodes must refine.
        """
        if type(set_of_selector_indexes) is not set:
            raise TypeError("set_of_selector_indexes must be a set.")
        return [descendant_node for descendant_node in self._descendant_nodes
                if set_of_selector_indexes < descendant_node.selector_indexes
                ]

    def add_descendant(self, descendant_node : 'BFinderNode'):
        """
        This method adds a descendant node to the current node.
        :param descendant_node: descendant node to be added.
        """
        if type(descendant_node) is not BFinderNode:
            raise TypeError("descendant_node must be a BFinderNode.")
        if descendant_node not in self._descendant_nodes:
            self._descendant_nodes.append(descendant_node)

