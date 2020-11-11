import functionality_new as funcs
import pandas as pd


def test_pareto():
    """ Test question 7 """

    final_result = pd.read_excel('test_data/Question_7.xlsx', 'Result')

    perc_users_income_threshold = funcs.pareto_proof_online_shop(data_sets=['test_data/Question_7.csv'], test=True)

    assert round(final_result.Final[0], 5) == round(perc_users_income_threshold, 5)
