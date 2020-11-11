import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


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
def plot_sold_product_category(data_sets,columns_used=('event_time','category_code','event_type'),chunk_size=1000000):
    '''
    This function return the number of sold product for category
    
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
            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')
            
            #clean our dataset
            chunk.dropna(subset=['category_code'],inplace=True)
            
            # filter the sold product
            chunk_purchase = chunk[chunk.event_type == 'purchase']

            # Introduce the column category
            chunk_purchase['category'] = chunk_purchase['category_code'].apply(lambda x: x.split('.')[0])
           
            # Convert event time into month
            chunk_purchase['event_time'] = chunk_purchase.event_time.dt.month

            chunk_list.append(chunk_purchase)

    working_data_set=pd.concat(chunk_list,ignore_index=True)

    # a plot showing the number of sold products per category
    working_data_set.groupby('event_time')['category'].value_counts().head(20).plot.bar( \
        figsize=(18, 7), \
        title='Top Category')
    plt.xlabel('Category')
    plt.ylabel('Number of sold products')
    plt.show()


def plot_visited_product_subcategory(data_sets,columns_used=('event_time','category_code','event_type','user_id'),chunk_size=1000000):
    '''
    This function return the number of visited product for subcategory
    
    '''
    processed_data_set = pd.DataFrame(columns=['event_time', 'subcategory_code', 'count_sub_categories'])
    processed_data_set.set_index(['event_time', 'subcategory_code'], inplace=True)

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
            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')
            
            #clean our dataset
            chunk=chunk[~chunk.category_code.isnull()]
            
            # filter the visited product
            chunk_view = chunk[chunk.event_type == 'view']
       
            # Introduce the column sub_category
            chunk_view['subcategory_code'] =chunk_view['category_code'].apply(
            lambda x: re.findall(r'\.(.*)', x)[0] if len(x.split('.')) != 1 else x)
            

            # Convert event time into month
            chunk_view['event_time'] = chunk_view.event_time.dt.month
            
            temp = chunk_view.groupby(['event_time', 'subcategory_code']).user_id.count().reset_index(name='count_sub_categories').set_index(['event_time', 'subcategory_code'])
            
            processed_data_set = temp.add(processed_data_set, fill_value=0)


    # A Plot showing the most visited subcategories
    plt.figure(figsize=(18, 7))
    processed_data_set.groupby(level=0, group_keys=False).apply(
        lambda x: x.sort_values(by=['count_sub_categories'], ascending=False).head(20)).plot.bar()
    plt.title('Top Subcategory', fontsize=18)
    plt.xlabel('Subcategory')
    plt.ylabel('Number of visited product')
    plt.show()


def ten_most_sold(data_sets,columns_used=('event_time','event_type','category_code','product_id','user_id'),chunk_size=1000000):
    '''
    This function returns the 10 most sold products for category
    
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
            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')
            
            #clean our dataset
            chunk=chunk[~chunk.category_code.isnull()]
            
            #filter the sold product
            chunk_purchase = chunk[chunk.event_type == 'purchase']
            
            # Introduce the column category
            chunk_purchase['category'] = chunk_purchase['category_code'].apply(lambda x: x.split('.')[0])
            
            #remove the column category_code that we don't need
            chunk_purchase.drop(columns=['category_code'],inplace=True)
            
            # Convert event time into month
            chunk_purchase['event_time'] = chunk_purchase.event_time.dt.month
            
            chunk_list.append(chunk_purchase)
            
    working_data_set=pd.concat(chunk_list,ignore_index=True)

    # find the category
    categorie = working_data_set['category'].unique()
    # find mounths
    mesi = working_data_set['event_time'].unique()

    # create a new dataframe which conteins for each product the number of sold pz
    df = working_data_set.groupby(['event_time', 'category', 'product_id']).count()
    df.reset_index(inplace=True)
    df.rename(columns={'user_id': 'totale_pezzi'}, inplace=True)
    df.drop(columns=['event_type'], inplace=True)

    # create a new dataframe which conteins mounth,category,product,number of sold pz
    new_df = pd.DataFrame(columns=df.columns)
    
    
    for m in mesi:
        for c in categorie:
            temp = df[(df['category'] == c) & (df['event_time'] == m)].sort_values('totale_pezzi',
                                                                                   ascending=False).head(10)
            new_df = pd.concat([new_df, temp])
    return new_df


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

#---------Research Question 4
def plot_profit_for_brand(data_sets,brand,columns_used=('event_time','brand','event_type','price'), chunk_size=1000000):
    '''
    This function return the plot of the profit of a brand passed in input
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
            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')

            #filter the sold product
            chunk_purchase = chunk[chunk.event_type == 'purchase']

            #filter the product sold by the brand in input
            chunk_purchase_brand=chunk_purchase[chunk_purchase['brand']==brand]

            # Convert event time into month
            chunk_purchase_brand['event_time'] = chunk_purchase_brand.event_time.dt.month

            #remove the columns we don't need
            chunk_purchase_brand.drop(columns=['brand', 'event_type'], inplace=True)

            chunk_list.append(chunk_purchase_brand)

    working_data_set=pd.concat(chunk_list,ignore_index=True)
 
    #plot the profit for each mounth
    #we use the yscale=log to be able to visualize the profit in a better way due to its size is of the order od 10^6
    plt.figure(figsize=(6,6))
    working_data_set.groupby('event_time').price.sum().plot.bar()
    plt.yscale('log')
    plt.show()
    
def plot_average_price_brand(data_sets,columns_used=('event_time','brand','event_type','price','product_id'), chunk_size=1000000):
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
            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')

            #consider the product view because these are the ones which are offered from the brand
            chunk_view=chunk[chunk['event_type']=='view']
    
            #remove the rows which contain a null value in brand
            chunk_view.dropna(subset=['brand'],inplace=True)
    
            #each product have to be considered only one time so we remove the duplicates
            chunk_view=chunk_view.drop_duplicates(subset=['product_id','price','brand'])
            #remove the columns we don't need
            chunk_view.drop(columns=['product_id', 'event_type'], inplace=True)

            chunk_list.append(chunk_view)

    working_data_set=pd.concat(chunk_list,ignore_index=True)


    plt.figure(figsize=(18,10))
    top_ten=working_data_set.groupby('brand').price.mean().sort_values(ascending=False).head(10)
    last_ten=working_data_set.groupby('brand').price.mean().sort_values(ascending=False).tail(10)
    final_brand=pd.concat((top_ten,last_ten))
    final_brand.plot.bar()
    plt.title('differences between average price of brand')
    plt.xlabel('brands')
    plt.ylabel('average price')
    plt.show()
    
# ---------Research Question 5
def plot_hourly_average_visitors(data_sets, day,columns_used=('event_time','event_type','user_id','price'), chunk_size=1000000):
    '''
    This function plot the hourly average visitors for a given day
    '''
    chunk_list = []
    
    assert isinstance(day,str), 'day must be provided as string'

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
            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')

            # filter the event visits
            chunk_view = chunk[chunk.event_type == 'view']

            #convert event time into month week day hour
            d = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
            
            chunk_view['day'] = chunk_view.event_time.dt.dayofweek
            chunk_view = chunk_view[chunk_view['day'] == d[day]]

            chunk_view['month'] = chunk_view.event_time.dt.month
            chunk_view['hour'] = chunk_view.event_time.dt.hour
            chunk_view['week']=chunk_view.event_time.dt.week

            #remove the columns we don't need
            chunk_view.drop(columns=['event_time', 'event_type'], inplace=True)

            chunk_view.drop(columns='day', inplace=True)

            chunk_list.append(chunk_view)

    working_data_set=pd.concat(chunk_list,ignore_index=True)

    # count the number of visitors for hour
    #working_data_set =working_data_set.groupby(['month', 'hour']).count()

    #working_data_set.rename(columns={'user_id': 'total_visits'}, inplace=True)

    #working_data_set.reset_index(inplace=True)

    # plot the averge visitors for hour for the day selectioned
    plt.figure(figsize=(6, 6))
    working_data_set.groupby(['month','wek','hour']).user_id.count().groupby('hour').mean().plot.bar()
    plt.xlabel('Hours')
    plt.ylabel('Average visitors')
    plt.title('Average visitors for hour for ' + day)
    plt.show()


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


def run_over_data_sets(data_sets, columns_used=('event_time', 'category_code', 'brand', 'event_type'), chunk_size=1000000):
    """
    This function will run over the data sets. Please provide the data set as a list of strings
    with the path in which the data set is located
    """

    # assert type(data_sets) == 'list', 'Data sets need to be provided as a list os strings with the path of the given
    # data sets'
    # assert type(data_sets[0]) == 'str', 'Elements of list need to be a string'

    # Define empty list in which we will append the chunked data sets
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

        # In order to avoid working with the complete data set we will split data in chunk sizes
        for chunk in month_data:
            # perform data filtering

            print('Column names:', chunk.columns.tolist())
            print('Length of chunk:', len(chunk))
            print(chunk.head(5))
            print('======================')

            # Any pre-processing action will be done here !!!!
            chunk_filter = chunk[chunk.event_type == 'purchase']
            # Any pre-processing action will be done here !!!!

            # Once the data filtering is done, append the chunk to list
            chunk_list.append(chunk_filter)

    working_data_set = pd.concat(chunk_list, ignore_index=True)

    return working_data_set
