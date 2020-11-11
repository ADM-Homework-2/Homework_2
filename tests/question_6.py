import functionality_new as funcs
import pandas as pd
import numpy as np


def test_conversion_rate():
    """ Test question 6 """

    expected_results = pd.read_excel('test_data/Question_6.xlsx', 'Results_1')

    python_results = funcs.conversion_rate(data_sets=['test_data/Question_6.csv'])

    assert np.round(expected_results['Conversion Rate'][0], 4) == python_results


def test_conversion_rate_per_category():
    """ Test question 6 """

    expected_results = pd.read_excel('test_data/Question_6.xlsx', 'Results_2')

    python_results = funcs.conversion_rate_per_category(data_sets=['test_data/Question_6.csv'])

    assert np.all(expected_results['Conversion Rate'] == python_results.conversion_rate.values)
    assert np.all(expected_results['category_name'] == python_results.index)
    assert np.all(expected_results['Count Purchase'] == python_results.purchase_count.values)

