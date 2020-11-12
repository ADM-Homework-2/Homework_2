import functionality_new as funcs
import pandas as pd
import numpy as np


def test_highest_price_per_category_brand():
    """ Test question 3 """

    expected_results = pd.read_excel('test_data/Question_3.xlsx', 'Results_1')

    _, python_results = funcs.category_brand_highest_price(data_sets=['test_data/Question_3.csv'],
                                                           missing_treatment='missing_')

    assert np.all(expected_results['Average per category'] == python_results.values)


def test_highest_price_per_brand_electronics():
    """ Test question 3 """

    expected_results = pd.read_excel('test_data/Question_3.xlsx', 'Results_2')

    python_results = funcs.plot_average_price_brand_category(data_sets=['test_data/Question_3.csv'],
                                                             missing_treatment='missing_',
                                                             test=True)

    assert np.all(expected_results['Average per brand'] == python_results.values)
    assert np.all(expected_results['Brand'] == python_results.index)


def test_plot_sold_product_category():
    """ test functionality where we are computing the number of sold products per category """

    expected_results = pd.read_excel('test_data/Question_2_1.xlsx', 'Results_1')

    python_results = funcs.plot_sold_product_category(data_sets=['test_data/Question_2_1.csv'])

    assert np.all(np.array(expected_results['Count']) == python_results.values)


def test_plot_visited_product_subcategory():
    """ test functionality where we compute the most visited sub_categories """

    expected_results = pd.read_excel('test_data/Question_2_2.xlsx', 'Results_1')

    python_results = funcs.plot_visited_product_subcategory(data_sets=['test_data/Question_2_2.csv'], chunk_size=5)

    assert np.all(np.array(expected_results['Count']) == np.array(python_results.count_sub_categories))