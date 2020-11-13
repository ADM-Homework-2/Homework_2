import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import progressbar
import seaborn as sns
pd.options.mode.chained_assignment = None


# ------- Research Question 1

# 1.a.

def customer_journey_funnel(data_set):
    """
    Show that the funnel in our online shop does not necesarily need to follow the steps view>>cart>>purchase
    
    :param data_set: Data set over which we will show this
    :return: Series with some examples
    """

    users_with_multiple_events = data_set.groupby('user_id').event_type.nunique()
    users_with_multiple_events = users_with_multiple_events[users_with_multiple_events > 1].index
    users_with_multiple_events_data_set = data_set[data_set.user_id.isin(users_with_multiple_events)]
    result = users_with_multiple_events_data_set.groupby(['user_id', 'user_session', 'event_type']).price.count()

    return result


def user_session_user_id(data_set):
    """
    Show that user_sessions have different user_ids in some cases

    :param data_set: Data set over which we will show this
    :return: Series with some examples
    """

    user_with_equal_user_sessions = data_set.groupby('user_session').user_id.nunique()
    users_sessions_with_multiple_user_ids = user_with_equal_user_sessions[user_with_equal_user_sessions > 1].index
    users_sessions_with_multiple_user_ids_data_set = data_set[
        data_set.user_session.isin(users_sessions_with_multiple_user_ids)]

    print('The percentage of errors is:',
          ((users_sessions_with_multiple_user_ids_data_set.user_id.nunique() / 2) / data_set.user_id.nunique()) * 100,
          ' %')

    return users_sessions_with_multiple_user_ids_data_set


def events_per_session(data_set):
    """
    Review what the most performed event is by average for each session

    :param data_set:
    :return: Plot with averages
    """

    # Given that there are some unique session_ids with different user_ids, we will groupby user session and user id
    session_df = data_set.user_session.nunique()
    session_number = len(session_df)
    event_df = data_set.groupby(['event_type'])['event_time'].count().reset_index(name="count")
    event_number = event_df['count']
    average_event = event_number / session_number

    # Plot average events done per session
    plt.figure(figsize=(12, 7))
    plt.hist(event_df['event_type'], weights=average_event)
    plt.title("Average number of events within a session", fontsize=25)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=20)
    plt.grid(alpha=0.8)


# 1.b.

# TODO: review the case in which for the same user and product we have view, view, view, cart, view, cart
def average_views_before_cart(data_set):
    """
    Compute the average times a user views a product before adding it to the cart

    :param data_set:
    :return: Integer with amount of views
    """

    # Since we are looking for the number of times before putting item into the cart we will only be interested in those
    # users and products for which the product was finally put into the cart
    cart_df = data_set[data_set['event_type'] == 'cart']

    # Merge by product id and user id, including the moment in the product was put in the cart
    cart_and_view_df = data_set[['event_time', 'event_type', 'user_id', 'product_id']].merge(cart_df[['event_time',
                                                                                                      'user_id',
                                                                                                      'product_id']],
                                                                                             how='right',
                                                                                             on=['product_id',
                                                                                                 'user_id'],
                                                                                             suffixes=('_view',
                                                                                                       '_cart'))

    # Keep only events for which the cart event came before the view event
    view_before_cart = cart_and_view_df[(cart_and_view_df['event_time_view'] < cart_and_view_df['event_time_cart']) & (
            cart_and_view_df['event_type'] == 'view')]

    # Count the number of views per user and per product and take the average over these values
    average_num_view = view_before_cart.groupby(
        ['user_id', 'product_id'])['event_time_view'].count().mean()

    return average_num_view


# 1.c.
# Given that the funnel does not apply in our online shop, we will consider the the probability as:
# prob = purchase, given that it was previously put in the cart/total products put in the cart
def probability_purchase_given_cart(data_set):
    """
    Compute the probability a product is purchased given it is sent to the cart

    :param data_set:
    :return: float
    """

    # Store all cart events
    cart_df = data_set[data_set['event_type'] == 'cart']

    # Join with complete data set. This way we have a data set with only the users and products that have ever been
    # stored in the cart, including the time in which they were put in the cart and the time in which any other event
    # occured (we will be interested in the purchase event time)
    cart_and_purchase = data_set[['event_time', 'event_type', 'user_id', 'product_id']].merge(
        cart_df[['event_time', 'user_id', 'product_id']], how='right',
        on=['product_id', 'user_id'], suffixes=('_purchase', '_cart'))

    # We are only interested in events which finally ended up in a purchase, and only in cases for which there was a
    # prior cart action
    cart_before_purchase = cart_and_purchase[
        (cart_and_purchase['event_time_cart'] < cart_and_purchase['event_time_purchase']) & (
                cart_and_purchase['event_type'] == 'purchase')]

    prob_purchase = len(cart_before_purchase) / len(cart_df)

    return prob_purchase


# 1.d.
# We don't have the event removefromcart. For this reason we will have to find some kind of approximation for the
# removefromcart event. We will assume that the user_session does not have memory, therefore everything that was in the
# cart once the session was ended and not bought will be removed from the cart. The next logical question would
# therefore be how to compute the moment in which the session ended. This will be done by looking at the time in which
# the last view happened

def average_time_remove_from_cart(data_set):
    """
    Compute the average time an item is in the cart (based on the first moment the product is put in the cart until the
    last view of a specific session)

    :param data_set:
    :return:
    """

    session_product_with_multiple_events = data_set.groupby(
        ['user_session', 'product_id']).event_type.nunique().reset_index(name='unique_event_type')
    session_product_with_2_different_events = session_product_with_multiple_events[
        session_product_with_multiple_events.unique_event_type == 2]
    # Make a dataframe with all user_sessions and products with 2 events
    merge_temp = session_product_with_2_different_events.merge(data_set, on=['user_session', 'product_id'])
    # From these users, remove the purchases
    merge_temp = merge_temp[merge_temp.event_type != 'purchase']
    session_product_with_view_and_cart = merge_temp.groupby(
        ['user_session', 'product_id']).event_type.nunique().reset_index(name='unique_event_type')
    session_product_with_view_and_cart = session_product_with_view_and_cart[
        session_product_with_view_and_cart.unique_event_type == 2]
    session_product_with_view_and_cart_data_set = session_product_with_view_and_cart.merge(data_set,
                                                                                           on=['user_session',
                                                                                               'product_id'])
    product_in_cart_df = session_product_with_view_and_cart_data_set[
        session_product_with_view_and_cart_data_set.event_type == 'cart'].groupby(
        ['user_session', 'product_id']).first()
    product_removed_from_cart_df = session_product_with_view_and_cart_data_set[
        session_product_with_view_and_cart_data_set.event_type == 'view'].groupby(['user_session', 'product_id']).last()

    product_in_cart = pd.to_datetime(product_in_cart_df['event_time'])
    product_removed_from_cart = pd.to_datetime(product_removed_from_cart_df['event_time'])

    time_in_cart = product_removed_from_cart - product_in_cart
    average_time_in_cart = time_in_cart.mean()

    return average_time_in_cart


# 1.e.

def average_time_between_view_cart_purchase(data_set):
    """
    Compute the average time between the first view and the insertion into the cart and insertion into the purchase
    (first occurrence in both cases)

    :param data_set:
    :return: Average time between view and cart, and view and purchase
    """

    # Average time between first view and when product was put into the cart for the first time
    cart_df = data_set[data_set['event_type'] == 'cart']
    common = data_set.merge(cart_df, on=['product_id', 'user_id'])
    common = common[common['event_type_y'] == 'cart']
    first_view_df = common.groupby(['product_id', 'user_id']).first()
    last_cart_df = common[common['event_type_x'] == 'cart'].groupby(['product_id', 'user_id']).first()

    first_view_df = pd.to_datetime(first_view_df['event_time_x'])
    last_cart_df = pd.to_datetime(last_cart_df['event_time_x'])

    time_before_cart = last_cart_df - first_view_df

    average_view_cart = time_before_cart.mean()

    # Average time between first view and when product was purchased for the first time
    purchase_df = data_set[data_set['event_type'] == 'purchase']
    common = data_set.merge(purchase_df, on=['product_id', 'user_id'])
    first_view_df = common.groupby(['product_id', 'user_id']).first()
    first_purchase_df = common[common['event_type_x'] == 'cart'].groupby(['product_id', 'user_id']).first()

    first_view_df = pd.to_datetime(first_view_df['event_time_x'])
    first_purchase_df = pd.to_datetime(first_purchase_df['event_time_x'])

    time_before_cart = first_purchase_df - first_view_df
    average_view_purchase = time_before_cart.mean()

    return average_view_cart, average_view_purchase


# -------- Research Question 2
def plot_sold_product_category(data_sets, columns_used=('event_time', 'category_code', 'event_type'),
                               chunk_size=1000000):
    '''
    This function return the number of sold product for category
    
    '''
    chunk_list = []
    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    for data_set in data_sets:
        print(data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 parse_dates=['event_time'],
                                 date_parser=pd.to_datetime,
                                 chunksize=chunk_size)
        for chunk in month_data:
            bar.update(i + 1)
            i += 1
            # clean our dataset
            chunk.dropna(subset=['category_code'], inplace=True)

            # filter the sold product
            chunk_purchase = chunk[chunk.event_type == 'purchase']

            # Introduce the column category
            chunk_purchase['category'] = chunk_purchase['category_code'].apply(lambda x: x.split('.')[0])

            # Convert event time into month
            chunk_purchase['event_time'] = chunk_purchase.event_time.dt.month

            chunk_list.append(chunk_purchase)

    bar.finish()
    print('Finished pre-processing data')

    working_data_set = pd.concat(chunk_list, ignore_index=True)

    # a plot showing the number of sold products per category
    final_data_set = working_data_set.groupby('event_time')['category'].value_counts()
    final_data_set = final_data_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(ascending=False).head(20))
    final_data_set = final_data_set.reset_index(name='Sold products per category')

    g = sns.catplot(x="category",
                    y="Sold products per category",
                    hue="event_time",
                    data=final_data_set,
                    kind="bar")
    g.set_xticklabels(rotation=90)

    return final_data_set


# TODO: Plot something nice
def plot_visited_product_subcategory(data_sets, columns_used=('event_time', 'category_code', 'event_type', 'user_id'),
                                     chunk_size=1000000):
    '''
    This function return the number of visited product for subcategory
    
    '''
    processed_data_set = pd.DataFrame(columns=['event_time', 'subcategory_code', 'count_sub_categories'])
    processed_data_set.set_index(['event_time', 'subcategory_code'], inplace=True)

    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    for data_set in data_sets:
        print(data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 parse_dates=['event_time'],
                                 date_parser=pd.to_datetime,
                                 chunksize=chunk_size)
        for chunk in month_data:
            bar.update(i + 1)
            i += 1
            # clean our dataset
            chunk = chunk[~chunk.category_code.isnull()]

            # filter the visited product
            chunk_view = chunk[chunk.event_type == 'view']

            # Introduce the column sub_category
            chunk_view['subcategory_code'] = chunk_view['category_code'].apply(
                lambda x: re.findall(r'\.(.*)', x)[0] if len(x.split('.')) != 1 else x)

            # Convert event time into month
            chunk_view['event_time'] = chunk_view.event_time.dt.month

            temp = chunk_view.groupby(['event_time', 'subcategory_code']).user_id.count().reset_index(
                name='count_sub_categories').set_index(['event_time', 'subcategory_code'])

            processed_data_set = temp.add(processed_data_set, fill_value=0)

    bar.finish()
    print('Finished pre-processing data')

    # A Plot showing the most visited subcategories for month
    processed_data_set = processed_data_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(by=['count_sub_categories'], ascending=False).head(20))
    processed_data_set.reset_index(inplace=True)
    g = sns.catplot(y='count_sub_categories',
                    x='subcategory_code',
                    hue='event_time',
                    data=processed_data_set,
                    kind='bar')
    g.set_xticklabels(rotation=90)

    return processed_data_set


def ten_most_sold(data_sets, columns_used=('event_time', 'event_type', 'category_code', 'product_id', 'user_id'),
                  chunk_size=1000000):
    '''
    This function returns the 10 most sold products for category
    
    '''
    chunk_list = []

    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    for data_set in data_sets:
        print(data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 parse_dates=['event_time'],
                                 date_parser=pd.to_datetime,
                                 chunksize=chunk_size)
        for chunk in month_data:
            bar.update(i + 1)
            i += 1
            # clean our dataset
            chunk = chunk[~chunk.category_code.isnull()]

            # filter the sold product
            chunk_purchase = chunk[chunk.event_type == 'purchase']

            # Introduce the column category
            chunk_purchase['category'] = chunk_purchase['category_code'].apply(lambda x: x.split('.')[0])

            # remove the column category_code that we don't need
            chunk_purchase.drop(columns=['category_code'], inplace=True)

            # Convert event time into month
            chunk_purchase['event_time'] = chunk_purchase.event_time.dt.month

            chunk_list.append(chunk_purchase)

    bar.finish()
    print('Finished pre-processing data')
    working_data_set = pd.concat(chunk_list, ignore_index=True)

    # find the category
    categories = working_data_set['category'].unique()
    # find months
    months = working_data_set['event_time'].unique()

    # create a new dataframe which contains for each product the number of sold products
    df = working_data_set.groupby(['event_time', 'category', 'product_id']).count()
    df.reset_index(inplace=True)
    df.rename(columns={'user_id': 'totale_pezzi'}, inplace=True)
    df.drop(columns=['event_type'], inplace=True)

    # create a new dataframe which contains month,category,product,number of sold pz
    new_df = pd.DataFrame(columns=df.columns)

    for m in months:
        for c in categories:
            temp = df[(df['category'] == c) & (df['event_time'] == m)].sort_values('totale_pezzi',
                                                                                   ascending=False).head(10)
            new_df = pd.concat([new_df, temp])

    return new_df


# ------- Research Question 3

def plot_average_price_brand_category(data_sets,
                                      columns_used=('category_code', 'brand', 'product_id', 'price'),
                                      category_name='electronics',
                                      missing_treatment='unknown_',
                                      num_top_brand=20,
                                      chunk_size=1000000,
                                      test=False):
    """
    This function will return the brand with the highest average price of their products per category

    :param data_sets: List with strings in which data sets are stored
    :param columns_used: Relevant columns for analysis
    :param category_name: Category we are interested in inspecting
    :param missing_treatment: If a variable is provided, the missings of brand and category code will be replaced with
        a new category name
    :param num_top_brand: Number of top brands we want to plot. Provide none if we want to see all (might lead to
        unreadable graphs)
    :param chunk_size: Chunks over which to read the data sets
    :param test: Boolean used for testing purposes
    :return: Plot
    """

    assert isinstance(data_sets,
                      list), 'Data sets need to be provided as a list os strings with the path of the given data sets'
    assert isinstance(data_sets[0], str), 'Elements of list need to be a string'

    # Define empty list in which we will append the chunked data sets
    chunk_list = []

    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    for data_set in data_sets:
        print('Running data_set:', data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 chunksize=chunk_size)

        # In order to avoid working with the complete data set we will split data in chunk sizes
        for chunk in month_data:
            bar.update(i + 1)
            i += 1
            # Pre-processing all data in the provided data frames
            # We will first replace all missing categories and brand with a own category. This will allow the online
            # shop to know whether unknown brands or categories have prices
            if missing_treatment:
                chunk['brand'] = chunk.brand.fillna(missing_treatment + 'brand')
                chunk['category_code'] = chunk.category_code.fillna(missing_treatment + 'brand')

            # Generate category name column
            chunk['parent_category'] = chunk['category_code'].apply(lambda x: x.split('.')[0])

            # Since we are only interested in specific categories we will filter out all irrelevant categories for now
            processed_chunk = chunk[chunk.parent_category == category_name]

            # Once the data filtering is done, append the chunk to list
            chunk_list.append(processed_chunk)

    bar.finish()
    working_data_set = pd.concat(chunk_list, ignore_index=True)
    print('Finished pre-processing data')

    # Unique prices can be identified by keys brand, product_id and price (we have seen that price vary during the same
    # month over the same product, and some brand names change over time with the product_id remaining the same)
    unique_prices = working_data_set.drop_duplicates(subset=['brand', 'product_id', 'price'])[['brand',
                                                                                               'product_id',
                                                                                               'price']]
    # Perform average over brand and product_id (to get the average price at product level), and then perform an
    # additional average at brand level to get the average price of each brand
    average_price = unique_prices.groupby(['brand', 'product_id']).price.mean().groupby(
        ['brand']).mean().sort_values(ascending=False)

    if test:
        return average_price

    if num_top_brand:
        average_price = average_price.head(num_top_brand)

    # Plot results
    average_price.plot.bar(figsize=(18, 6))
    plt.title(category_name, fontsize=18)
    plt.xlabel('Brand Name')
    plt.ylabel('Average Product Price of Brand (€)')
    plt.show()


def category_brand_highest_price(data_sets,
                                 columns_used=('category_code', 'brand', 'product_id', 'price'),
                                 missing_treatment='unknown_',
                                 chunk_size=1000000):
    """
    This function will run over the data sets. Please provide the data set as a list of strings
    with the path in which the data set is located
    """

    assert isinstance(data_sets,
                      list), 'Data sets need to be provided as a list os strings with the path of the given data sets'
    assert isinstance(data_sets[0], str), 'Elements of list need to be a string'

    # Define empty list in which we will append the chunked data sets
    chunk_list = []
    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    for data_set in data_sets:
        print('Running data_set:', data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 chunksize=chunk_size)

        # In order to avoid working with the complete data set we will split data in chunk sizes
        for chunk in month_data:
            bar.update(i + 1)
            i += 1
            # Any pre-processing action will be done here
            # Following a similar logic as before (with the only difference that we will now not filter by category)
            if missing_treatment:
                chunk['brand'] = chunk.brand.fillna(missing_treatment + 'brand')
                chunk['category_code'] = chunk.category_code.fillna(missing_treatment + 'brand')

            # Generate category name column
            chunk['parent_category'] = chunk['category_code'].apply(lambda x: x.split('.')[0])

            # Remove duplicates, such that we keep unique prices
            processed_chunk = chunk.drop_duplicates(subset=['parent_category', 'brand', 'product_id', 'price'])[
                ['parent_category', 'brand', 'product_id', 'price']]

            # Once the data processing is done, append the chunk to list
            chunk_list.append(processed_chunk)

    bar.finish()
    working_data_set = pd.concat(chunk_list, ignore_index=True)
    print('Finished pre-processing data')

    # Make sure we really have the unique price values at category, brand and product_id levels over the different
    # chunks
    unique_prices = working_data_set.drop_duplicates(subset=['parent_category', 'brand', 'product_id', 'price'])[
        ['parent_category', 'brand', 'product_id', 'price']]

    # Group to obtain the average product price at category level for each brand
    grouped_set = unique_prices.groupby(['parent_category', 'brand', 'product_id']).price.mean().groupby(
        ['parent_category', 'brand']).mean()

    # Sort average price of brands at category level
    sorted_data_frame = grouped_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(ascending=False))

    # Keep the brand with the highest average price
    highest_price_brands = grouped_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(ascending=False).head(1)).sort_values(ascending=False)

    return sorted_data_frame, highest_price_brands


# ---------Research Question 4
def plot_profit_for_brand(data_sets, brand, columns_used=('event_time', 'brand', 'event_type', 'price'),
                          chunk_size=1000000):
    '''
    This function return the plot of the profit of a brand passed in input
    '''

    processed_data_set = pd.DataFrame(columns=['event_time', 'brand', 'total_profit'])
    processed_data_set.set_index(['event_time', 'brand'], inplace=True)

    #brands_set = set()
    for data_set in data_sets:
        print(data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 parse_dates=['event_time'],
                                 date_parser=pd.to_datetime,
                                 chunksize=chunk_size)
        for chunk in month_data:

            # filter the sold product
            chunk_purchase = chunk[chunk.event_type == 'purchase']

            # clean the column brand
            chunk_purchase.dropna(subset=['brand'], inplace=True)

            # find all brands
            # brands_array = chunk_purchase['brand'].unique()
            # for b in brands_array:
            #     brands_set.add(b)

            # Convert event time into month
            chunk_purchase['event_time'] = chunk_purchase.event_time.dt.month

            # remove the columns we don't need
            chunk_purchase.drop(columns=['event_type'], inplace=True)

            # calculate the profit for each brand
            chunk_purchase = chunk_purchase.groupby(['event_time', 'brand']).price.sum().reset_index(
                name='total_profit').set_index(['event_time', 'brand'])

            processed_data_set = chunk_purchase.add(processed_data_set, fill_value=0)

    # to be able to do some filter we have to restore the columns
    processed_data_set.reset_index(inplace=True)

    # Compute the unique brands
    brands_set = processed_data_set.brand.unique().tolist()

    # plot the profit for month for the brand given in input
    g = processed_data_set[processed_data_set['brand'] == brand]
    g.plot(x='event_time', y='total_profit', kind='bar')
    plt.show()

    # Data set in which we will store all relevant information for final analysis
    final_data_set = pd.DataFrame(columns=['brand', 'Max Loss between months', 'Months of loss'])

    dict_temp = {'10': 'October',
                 '11': 'November',
                 '12': 'December',
                 '1': 'January',
                 '2': 'February',
                 '3': 'March',
                 '4': 'April'}

    # We will loop over all brands computing its max loss values
    for b in brands_set:
        # Get the brand with its profit per month
        temp = processed_data_set[processed_data_set[
                                      'brand'] == b]
        profits = np.array(temp['total_profit'])
        months = np.array(temp['event_time'])
        if len(temp) == 1:
            continue
        else:
            # If brand appears in several months
            max_loss = np.min(np.diff(profits))
            months_of_loss_temp = np.argmin(np.diff(profits))
            months_of_loss = dict_temp[str(months[months_of_loss_temp])] + ' to ' + dict_temp[str(months[months_of_loss_temp + 1])]

        final_data_set = final_data_set.append(pd.Series([b, max_loss, months_of_loss],
                                               index=final_data_set.columns), ignore_index=True)

    # return the 3 brand with biggest losses in earnings
    final_data_set = final_data_set.sort_values(by='Max Loss between months')

    return final_data_set


def plot_average_price_brand(data_sets, columns_used=('event_time', 'brand', 'price', 'product_id'),
                             chunk_size=1000000):
    '''
    This function return the plot of the average price of products of different brands
    '''
    chunk_list = []

    for data_set in data_sets:
        print(data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 parse_dates=['event_time'],
                                 date_parser=pd.to_datetime,
                                 chunksize=chunk_size)
        for chunk in month_data:

            # consider the product view because these are the ones which are offered from the brand
            # chunk_view = chunk[chunk['event_type'] == 'view']

            # remove the rows which contain a null value in brand
            chunk.dropna(subset=['brand'], inplace=True)

            # each product have to be considered only one time so we remove the duplicates
            chunk_view = chunk.drop_duplicates(subset=['product_id', 'price', 'brand'])
            # remove the columns we don't need
            chunk_view.drop(columns=['product_id'], inplace=True)

            chunk_list.append(chunk_view)

    working_data_set = pd.concat(chunk_list, ignore_index=True)

    plt.figure(figsize=(18, 10))
    grouped_data_set = working_data_set.groupby('brand').price.mean().sort_values(ascending=False)
    top_ten = grouped_data_set.head(10)
    last_ten = grouped_data_set.tail(10)
    final_brand = pd.concat((top_ten, last_ten))
    final_brand.plot.bar()
    plt.title('differences between average price of brand')
    plt.xlabel('brands')
    plt.ylabel('average price')
    plt.show()


# ---------Research Question 5
def plot_hourly_average_visitors(data_sets, day, columns_used=('event_time', 'event_type', 'user_id', 'price'),
                                 chunk_size=1000000):
    '''
    This function plot the hourly average visitors for a given day
    '''
    chunk_list = []

    assert isinstance(day, str), 'day must be provided as string'

    for data_set in data_sets:
        print(data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 parse_dates=['event_time'],
                                 date_parser=pd.to_datetime,
                                 chunksize=chunk_size)
        for chunk in month_data:

            # filter the event visits
            chunk_view = chunk[chunk.event_type == 'view']

            # convert event time into month week day hour
            d = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

            chunk_view['day'] = chunk_view.event_time.dt.dayofweek
            chunk_view = chunk_view[chunk_view['day'] == d[day]]

            chunk_view['month'] = chunk_view.event_time.dt.month
            chunk_view['hour'] = chunk_view.event_time.dt.hour
            chunk_view['week'] = chunk_view.event_time.dt.week

            # remove the columns we don't need
            chunk_view.drop(columns=['event_time', 'event_type'], inplace=True)

            chunk_view.drop(columns='day', inplace=True)

            chunk_list.append(chunk_view)

    working_data_set = pd.concat(chunk_list, ignore_index=True)

    # plot the average visitors for hour for the day selected
    plt.figure(figsize=(6, 6))
    final_data_set = working_data_set.groupby(['month', 'week', 'hour']).user_id.count().groupby('hour').mean()
    final_data_set.plot.bar()
    plt.xlabel('Hours')
    plt.ylabel('Average visitors')
    plt.title('Average visitors for hour for ' + day)
    plt.show()

    return final_data_set


# ------- Research Question 6

def conversion_rate(data_sets,
                    chunk_size=1000000):
    """
    Purchases against views conversion rate

    :param data_sets: List with strings in which data sets are stored
    :param chunk_size: Chunks over which to read the data sets
    :return: Conversion rate of online shop
    """

    assert isinstance(data_sets,
                      list), 'Data sets need to be provided as a list os strings with the path of the given data sets'
    assert isinstance(data_sets[0], str), 'Elements of list need to be a string'

    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    # Define integer with number of views and number of purchases
    number_views = 0
    number_purchases = 0
    for data_set in data_sets:
        print('Running data_set:', data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=['event_type'],
                                 encoding="ISO-8859-1",
                                 chunksize=chunk_size)

        # In order to avoid working with the complete data set we will split data in chunk sizes
        for chunk in month_data:
            bar.update(i + 1)
            i += 1

            # Look at the number of purchases and views in each chunk of data and include them in the corresponding
            # variable
            number_views += len(chunk[chunk.event_type == 'view'])
            number_purchases += len(chunk[chunk.event_type == 'purchase'])

    bar.finish()

    overall_conversion_rate = number_purchases / number_views

    return round(overall_conversion_rate, 4)


def conversion_rate_per_category(data_sets,
                                 columns_used=('category_code', 'event_type'),
                                 missing_treatment='unknown_category',
                                 chunk_size=1000000):
    """
    Compute the conversion rate for each category and plot the results in decreasing order

    :param data_sets: List with strings in which data sets are stored
    :param columns_used: Columns required for analysis
    :param chunk_size: Chunks over which to read the data sets
    :return: Conversion rate of online shop
    """

    assert isinstance(data_sets,
                      list), 'Data sets need to be provided as a list os strings with the path of the given data sets'
    assert isinstance(data_sets[0], str), 'Elements of list need to be a string'

    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0
    # Define data frame where we will store number of views and number of purchases
    number_views_per_category = pd.DataFrame(columns=['category_id', 'view_count'])
    number_views_per_category.set_index('category_id', inplace=True)

    number_purchases_per_category = pd.DataFrame(columns=['category_id', 'purchase_count'])
    number_purchases_per_category.set_index('category_id', inplace=True)

    for data_set in data_sets:
        print('Running data_set:', data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 chunksize=chunk_size)

        # In order to avoid working with the complete data set we will split data in chunk sizes
        for chunk in month_data:
            bar.update(i + 1)
            i += 1

            # Replace missings
            if missing_treatment:
                chunk['category_code'] = chunk.category_code.fillna(missing_treatment)

            # Generate category name column
            chunk['parent_category'] = chunk['category_code'].apply(lambda x: x.split('.')[0])

            # Look at the number of purchases and views in each chunk of data. Set category name into the index so that
            # we can then add dataframes in an easy manner
            processed_chunk_view = chunk[chunk.event_type == 'view'].groupby(
                'parent_category').event_type.count().reset_index(name='view_count').set_index(['parent_category'])
            processed_chunk_purchase = chunk[chunk.event_type == 'purchase'].groupby(
                'parent_category').event_type.count().reset_index(name='purchase_count').set_index(['parent_category'])

            # Make something such that we can sum processed_chunk_view / processed_chunk_purchase over all chunks (we
            # can sum easily indexed dataframes)
            number_views_per_category = processed_chunk_view.add(number_views_per_category, fill_value=0)
            number_purchases_per_category = processed_chunk_purchase.add(number_purchases_per_category, fill_value=0)

    bar.finish()

    # Merge processed_chunk_view and processed_chunk_purchase and do purchase/view. With this dataframe we can do plot
    final_data_set = pd.concat([number_views_per_category, number_purchases_per_category], axis=1).fillna(0)

    final_data_set['conversion_rate'] = final_data_set.purchase_count / final_data_set.view_count
    final_data_set = final_data_set.sort_values(by=['conversion_rate'], ascending=False)

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(final_data_set.index, final_data_set.purchase_count, color="red")
    ax.set_xlabel("Categories", fontsize=14)
    plt.xticks(rotation=45)
    ax.set_ylabel("Number of purchases", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(final_data_set.index, final_data_set.conversion_rate, color="blue")
    ax2.set_ylabel("Conversion Rate", fontsize=14)
    plt.show()

    return final_data_set


# ------- Research Question 7

def pareto_proof_online_shop(data_sets,
                             columns_used=('event_type', 'user_id', 'price'),
                             chunk_size=1000000,
                             income_threshold=0.8,
                             test=False):
    """
    Prove that the pareto principle is fulfilled by our online shop

    :param data_sets: List with strings in which data sets are stored
    :param columns_used: Columns required for analysis
    :param chunk_size: Chunks over which to read the data sets
    :param income_threshold: Threshold for which we will prove the amount of customers that produce that amount of
        income
    :param test: Boolean used for testing purposes
    :return: Plot and percentage of customers that generate the given income_threshold
    """

    assert isinstance(data_sets,
                      list), 'Data sets need to be provided as a list os strings with the path of the given data sets'
    assert isinstance(data_sets[0], str), 'Elements of list need to be a string'

    bar = progressbar.ProgressBar(maxval=415,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()
    i = 0

    # Define empty list in which we will append the chunked data sets
    chunk_list = []

    # Run over different data sets
    for data_set in data_sets:
        print('Running data_set:', data_set)
        month_data = pd.read_csv(data_set, sep=',',
                                 delimiter=None, header='infer',
                                 usecols=columns_used,
                                 encoding="ISO-8859-1",
                                 chunksize=chunk_size)

        # In order to avoid working with the complete data set we will split data in chunk sizes
        for chunk in month_data:
            bar.update(i + 1)
            i += 1

            # We are only interested in purchase events
            processed_chunk = chunk[chunk.event_type == 'purchase']

            # Once the data filtering is done, append the chunk to list
            chunk_list.append(processed_chunk)

    bar.finish()
    working_data_set = pd.concat(chunk_list, ignore_index=True)

    # Compute the income at user_id level
    income_values = working_data_set.groupby(working_data_set.user_id).price.sum().sort_values(ascending=False).values

    # Compute the percentage of income each user generates
    income_perc_values = income_values / sum(income_values)

    # Compute the cumulative income percentage and the percentage of users in an array (both arrays ten to 1)
    income_percentage_cumulative = np.cumsum(income_perc_values)
    users_percentage = np.linspace(1 / len(income_values), 1, len(income_values))

    # Compute the percentage of users that generate the percentage of threshold income (used for plotting purposes)
    perc_income_threshold = income_percentage_cumulative[income_percentage_cumulative < income_threshold]
    perc_users_income_threshold = len(perc_income_threshold) / len(income_percentage_cumulative)

    if test:
        return perc_users_income_threshold

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
