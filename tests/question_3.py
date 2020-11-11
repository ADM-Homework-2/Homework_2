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


