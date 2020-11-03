import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------- Research Question 3

def plot_average_price_brand_category(data_set, category_id, missing_treatment='unknown_brand'):
    """
    This function will return the brand with the highest average price of their products per category

    :param data_set: Data set over which we will do the analysis
    :param category_id: Category ID we will be plotting
    :param missing_treatment: Category in which we will convert all missing brand observations
    :return: Plot
    """

    if missing_treatment:
        data_set['brand'] = data_set.brand.fillna(missing_treatment)

    data_set['event_time_month'] = pd.to_datetime(data_set.event_time).dt.month

    # make sure that price is unique at product_id and event_time.dt.month level
    assert type(category_id) == int, 'Category ID needs to be an integer'

    data_set = data_set[data_set['category_id'] == category_id]

    full_data_set_unique_prices = data_set.drop_duplicates(subset=['brand', 'product_id',
                                                                   'price'])[['brand',
                                                                              'product_id',
                                                                              'price']]
    average_price = full_data_set_unique_prices.groupby(['brand', 'product_id']).price.mean().groupby(
        ['brand']).mean().sort_values(ascending=False)

    average_price.plot.bar(figsize=(18, 6))
    category_code_value = data_set.category_code.iloc[0]
    if type(category_code_value) == str:
        plt.title('Category ' + category_code_value, fontsize=18)
    else:
        plt.title('Category ' + str(category_id), fontsize=18)
    plt.xlabel('Brand Name')
    plt.ylabel('Average Product Price of Brand (€)')
    plt.show()


def category_brand_highest_price(data_set, missing_treatment='unknown_brand'):
    """
    This function will return the brand with the highest average price of their products per category

    :param data_set: Data set over which we will do the analysis
    :param missing_treatment: Category in which we will convert all missing brand observations
    :return: 2 Series, one with all observations ordered and another with the highest brand values
    """

    if missing_treatment:
        data_set['brand'] = data_set.brand.fillna(missing_treatment)

    # Convert event time into month
    data_set['event_time_month'] = pd.to_datetime(data_set.event_time).dt.month

    # make sure that price is unique at product_id and event_time.dt.month level
    # TODO: include some quick check

    # Remove duplicates (review Jupyter notebook for detailed rational)
    full_data_set_unique_prices = data_set.drop_duplicates(subset=['category_id', 'brand', 'product_id',
                                                                   'price'])[['category_id', 'brand',
                                                                              'product_id',
                                                                              'price']]
    # Do average price over different time points (first groupby) and then apply price average over all products
    # (second groupby)
    grouped_set = full_data_set_unique_prices.groupby(['category_id', 'brand', 'product_id']).price.mean().groupby(
        ['category_id', 'brand']).mean()

    # Sort average price of brands at category level
    sorted_data_frame = grouped_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(ascending=False))

    # Keep the brand with the highest average price
    highest_price_brands = grouped_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(ascending=False).head(1))

    return sorted_data_frame, highest_price_brands


# ------- Research Question 6


def conversion_rate(data_set, nominator='purchase', denominator='view'):
    """
    Purchases against views conversion rate

    :param data_set: Data set over which the overall conversion rate will be computed
    :param nominator: Number of conversions (default is purchase)
    :param denominator: Number of possible conversions (default is view)
    :return: Conversion rate
    """

    rate = len(data_set[data_set.event_type == nominator]) / len(data_set[data_set.event_type == denominator])

    return round(rate, 4)


# TODO: make sure we are understanding what we are being requested
def conversion_rate_per_category(data_set):
    """
    Compute the conversion rate for each category and plot the results in decreasing order

    :param data_set: Data set over which analysis will be performed
    :return: Plot conversion rate and return data frame with the corresponding results
    """

    views_per_category = data_set[data_set.event_type == 'view'].groupby(['category_id']).agg(
        num_view=('price', 'count'))
    purchases_per_category = data_set[data_set.event_type == 'purchase'].groupby(['category_id']).agg(
        num_purchase=('price', 'count'))

    view_purchase_per_category = views_per_category.merge(purchases_per_category, how='outer', left_index=True,
                                                          right_index=True)
    view_purchase_per_category['purchase_rate'] = (view_purchase_per_category.num_purchase /
                                                   view_purchase_per_category.num_view) * 100
    view_purchase_per_category = view_purchase_per_category.sort_values(by='purchase_rate', ascending=False)[
        'purchase_rate'].fillna(0)

    view_purchase_per_category.plot.bar(figsize=(18, 6))
    plt.title('Purchase Rate Per Category', fontsize=18)
    plt.xlabel('Category ID')
    plt.ylabel('Purchase Rate (%)')
    plt.show()

    return view_purchase_per_category


# ------- Research Question 7

def pareto_proof_online_shop(data_set, income_threshold=0.8):
    """
    Prove that the pareto principle is fulfilled by our online shop

    :param data_set: Data set over which pareto principle wil be proven
    :param income_threshold: Threshold for which we will prove the amount of customers that produce that amount of
        income
    :return: Plot and percentage of customers that generate the given income_threshold
    """

    # Ccompute the income at user_id level
    purchase_events = data_set[data_set.event_type == 'purchase']
    income_per_user_id = purchase_events.groupby(purchase_events.user_id).price.sum().sort_values(ascending=False)

    # From the previous Series extract the income and users that have generated the total income
    income_values = income_per_user_id.values
    users = income_per_user_id.index

    # Construct cumulative percentage of income and users
    income_percentage_cumulative = [income_values[0]] * len(income_values)
    for i in range(1, len(income_percentage_cumulative)):
        income_percentage_cumulative[i] = income_percentage_cumulative[i - 1] + income_values[i]
    income_percentage_cumulative = income_percentage_cumulative / np.sum(income_values)
    users_percentage = np.linspace(1 / len(income_values), 1, len(income_values))

    # Compute the percentage of users that generate the percentage of threshold income (used for plotting purposes)
    perc_users_income_threshold = users_percentage[np.argmax(income_percentage_cumulative > income_threshold)]

    plt.figure(figsize=(8, 6))
    plt.plot(income_percentage_cumulative, users_percentage)
    plt.title('Pareto Proof for Online Shop')
    plt.xlabel('Cumulative Income (%)')
    plt.ylabel('Cumulative Users (%)')
    plt.plot([income_threshold, income_threshold], [0, perc_users_income_threshold], color='red',
             linewidth=1.5, linestyle="--")
    plt.plot([0, income_threshold], [perc_users_income_threshold, perc_users_income_threshold], color='red',
             linewidth=1.5, linestyle="--")
    plt.scatter([income_threshold, ], [perc_users_income_threshold, ], 50, color='red')
    plt.annotate(
        '(' + str(round(income_threshold * 100, 1)) + '%, ' + str(round(perc_users_income_threshold * 100, 1)) + '%)',
        xy=(income_threshold, perc_users_income_threshold), xycoords='data',
        xytext=(-80, +30), textcoords='offset points', fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()

