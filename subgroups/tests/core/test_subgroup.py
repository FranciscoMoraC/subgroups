# -*- coding: utf-8 -*-

# Contributors:
#    Antonio López Martínez-Carrasco <antoniolopezmc1995@gmail.com>

"""Tests of the functionality contained in the file 'core/subgroup.py'.
"""

from subgroups.core.pattern import Pattern
from subgroups.core.selector import Selector
from subgroups.core.subgroup import Subgroup
from subgroups.core.operator import Operator
from pandas import DataFrame
import unittest

class TestSubgroup(unittest.TestCase):

    def test_Subgroup_general(self):
        subgroup1 = Subgroup.generate_from_str("Description: [], Target: age = 34")
        self.assertEqual(subgroup1.description, Pattern([]))
        self.assertEqual(subgroup1.target, Selector.generate_from_str("age = 34"))
        subgroup2 = Subgroup.generate_from_str("Description: [at1 = 'a', at2 >= 78], Target: age = 34")
        subgroup3 = subgroup2.copy()
        self.assertNotEqual(id(subgroup2), id(subgroup3))
        self.assertNotEqual(id(subgroup2.description), id(subgroup3.description))
        self.assertNotEqual(id(subgroup2.description._list_of_selectors), id(subgroup3.description._list_of_selectors))
        self.assertEqual(id(subgroup2.target), id(subgroup3.target))
        self.assertIsNot(subgroup2, subgroup3)
        self.assertIsNot(subgroup2.description, subgroup3.description)
        self.assertIsNot(subgroup2.description._list_of_selectors, subgroup3.description._list_of_selectors)
        self.assertIs(subgroup2.target, subgroup3.target)
        subgroup4 = Subgroup(Pattern([Selector("at1", Operator.EQUAL, 'a'), Selector("at2", Operator.GREATER_OR_EQUAL, 78)]), Selector("age", Operator.EQUAL, 34))
        self.assertNotEqual(subgroup1, subgroup2)
        self.assertEqual(subgroup2, subgroup4)
        self.assertEqual(subgroup3, subgroup4)
        self.assertEqual(str(Subgroup(Pattern([Selector("a", Operator.NOT_EQUAL, 'value'), Selector("b", Operator.GREATER_OR_EQUAL, 78)]), Selector("age", Operator.LESS, 34))), str(Subgroup.generate_from_str("Description: [a != value, b >= 78], Target: age < 34")))

    def test_Subgroup_filter(self):
        df1 = DataFrame({"a" : [1,5,6,9,8,7,4,1,2,3], "b" : [55,69,85,57,51,78,98,56,54,85], "target" : [0,1,0,1,1,0,0,0,1,0]})
        df1_filtered_use_description_use_target = DataFrame({"a" : [8], "b" : [51], "target" : [1]}, index=[4])
        df1_filtered_use_description = DataFrame({"a" : [8], "b" : [51], "target" : [1]}, index=[4])
        df1_filtered_use_target = DataFrame({"a" : [5,9,8,2], "b" : [69,57,51,54], "target" : [1,1,1,1]}, index=[1,3,4,8])
        subgroup1 = Subgroup(Pattern.generate_from_str("[a > 5, b < 55]"), Selector("target", Operator.EQUAL, 1))
        filter1_subgroup1_df1 = subgroup1.filter(df1, use_description=True, use_target=True)
        self.assertIs(type(filter1_subgroup1_df1), tuple)
        filter2_subgroup1_df1 = subgroup1.filter(df1, use_description=True, use_target=False)
        self.assertIs(type(filter2_subgroup1_df1), tuple)
        filter3_subgroup1_df1 = subgroup1.filter(df1, use_description=False, use_target=True)
        self.assertIs(type(filter3_subgroup1_df1), tuple)
        filter4_subgroup1_df1 = subgroup1.filter(df1, use_description=False, use_target=False)
        self.assertIs(type(filter4_subgroup1_df1), tuple)
        self.assertTrue((filter1_subgroup1_df1[0] == df1_filtered_use_description_use_target).all().all())
        self.assertTrue((filter2_subgroup1_df1[0] == df1_filtered_use_description).all().all())
        self.assertTrue((filter3_subgroup1_df1[0] == df1_filtered_use_target).all().all())
        self.assertTrue((filter4_subgroup1_df1[0] == df1).all().all())
        self.assertEqual(filter1_subgroup1_df1[1], 1)
        self.assertEqual(filter2_subgroup1_df1[1], 1)
        self.assertEqual(filter3_subgroup1_df1[1], 1)
        self.assertEqual(filter4_subgroup1_df1[1], 1)
        self.assertEqual(filter1_subgroup1_df1[2], 0)
        self.assertEqual(filter2_subgroup1_df1[2], 0)
        self.assertEqual(filter3_subgroup1_df1[2], 0)
        self.assertEqual(filter4_subgroup1_df1[2], 0)
        self.assertEqual(filter1_subgroup1_df1[3], 4)
        self.assertEqual(filter2_subgroup1_df1[3], 4)
        self.assertEqual(filter3_subgroup1_df1[3], 4)
        self.assertEqual(filter4_subgroup1_df1[3], 4)
        self.assertEqual(filter1_subgroup1_df1[4], 6)
        self.assertEqual(filter2_subgroup1_df1[4], 6)
        self.assertEqual(filter3_subgroup1_df1[4], 6)
        self.assertEqual(filter4_subgroup1_df1[4], 6)
        df2 = DataFrame({"a" : [9,5,6,9,8,7,4,1,2,3], "b" : [50,69,85,57,51,78,98,56,54,85], "target" : [0,1,0,1,1,0,0,0,1,0]})
        df2_filtered_use_description_use_target = DataFrame({"a" : [8], "b" : [51], "target" : [1]}, index=[4])
        df2_filtered_use_description = DataFrame({"a" : [9,8], "b" : [50,51], "target" : [0,1]}, index=[0,4])
        df2_filtered_use_target = DataFrame({"a" : [5,9,8,2], "b" : [69,57,51,54], "target" : [1,1,1,1]}, index=[1,3,4,8])
        filter1_subgroup1_df2 = subgroup1.filter(df2, use_description=True, use_target=True)
        filter2_subgroup1_df2 = subgroup1.filter(df2, use_description=True, use_target=False)
        filter3_subgroup1_df2 = subgroup1.filter(df2, use_description=False, use_target=True)
        filter4_subgroup1_df2 = subgroup1.filter(df2, use_description=False, use_target=False)
        self.assertTrue((filter1_subgroup1_df2[0] == df2_filtered_use_description_use_target).all().all())
        self.assertTrue((filter2_subgroup1_df2[0] == df2_filtered_use_description).all().all())
        self.assertTrue((filter3_subgroup1_df2[0] == df2_filtered_use_target).all().all())
        self.assertTrue((filter4_subgroup1_df2[0] == df2).all().all())
        self.assertEqual(filter1_subgroup1_df2[1], 1)
        self.assertEqual(filter2_subgroup1_df2[1], 1)
        self.assertEqual(filter3_subgroup1_df2[1], 1)
        self.assertEqual(filter4_subgroup1_df2[1], 1)
        self.assertEqual(filter1_subgroup1_df2[2], 1)
        self.assertEqual(filter2_subgroup1_df2[2], 1)
        self.assertEqual(filter3_subgroup1_df2[2], 1)
        self.assertEqual(filter4_subgroup1_df2[2], 1)
        self.assertEqual(filter1_subgroup1_df2[3], 4)
        self.assertEqual(filter2_subgroup1_df2[3], 4)
        self.assertEqual(filter3_subgroup1_df2[3], 4)
        self.assertEqual(filter4_subgroup1_df2[3], 4)
        self.assertEqual(filter1_subgroup1_df2[4], 6)
        self.assertEqual(filter2_subgroup1_df2[4], 6)
        self.assertEqual(filter3_subgroup1_df2[4], 6)
        self.assertEqual(filter4_subgroup1_df2[4], 6)
