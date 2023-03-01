# all imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    laptop_data_path = 'C:/Users/Eo/Desktop/Coding/laptop_vis/Cleaned_Laptop_data.csv'
    laptop_data = pd.read_csv(laptop_data_path)

    # look at data
    print(laptop_data.columns)

    '''
    Preliminary analysis
    From personal knowledge:
    HIGH IMPORTANCE: brand, ram_gb, ssd, graphic_card_gb
    We can double check by analyzing correlation with latest_price
    '''

    # dataset taken from kaggle has prices listed as Indian Rupee
    # as of Feb 27, 2023, 1 Indian Rupee = 0.67 Philippine Peso
    # so multiply latest_price and old_price columns by scalar 0.67

    laptop_data['latest_price'] = laptop_data['latest_price'].apply(lambda x: x * 0.67)
    laptop_data['old_price'] = laptop_data['old_price'].apply(lambda x: x * 0.67)

    # check
    print(laptop_data['latest_price'].describe())

    # min is now PHP 9373.30 and max is now PHP 296133.30
    # make bounds 0 - 350000 for the graphs

    # histogram
    sns.distplot(laptop_data['latest_price'])

    # scatter plots with brand, ram_gb, ssd, and graphic_card_gb
    # ram_gb, ssd, and graphic_card_gb are numerical, but only have
    # a few bins so can be lumped as str type for the plots
    maybe_important = ['brand', 'ram_gb', 'ssd', 'graphic_card_gb']
    for feature in maybe_important:
        laptop_data[feature] = laptop_data[feature].astype(str)
        data = pd.concat([laptop_data['latest_price'], laptop_data[feature]], axis = 1)
        data.plot.scatter(x = feature, y = 'latest_price', ylim = (0, 350000))
        plt.xticks(rotation = 90)

    # some of the ram_gb entries are NAN or not a conventional number
    # will have to decide if outliers should be removed

    feature = 'brand'
    data = pd.concat([laptop_data['latest_price'], laptop_data[feature]], axis = 1)
    f, ax = plt.subplots(figsize = (8, 6))
    fig = sns.boxplot(x = feature, y = 'latest_price', data = laptop_data)
    plt.xticks(rotation=90)
    fig.axis(ymin = 0, ymax = 350000)

    '''
    CONCLUSIONS:
    Alienware, Apple and Lenovo brand laptops in that order are the most expensive
    Since there are some brands with only 4-5 representatives, may be best to remove them
        so the model learns better
    May also be wise to remove the non-standard RAM values by imputing
    '''

    # correlation matrix
    corrmat = laptop_data.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax = 0.8, annot = True, square = True, xticklabels = 1, yticklabels = 1)

    '''
    Naturally, the group of old_price, latest_price, and discount are heavily correlated
    with each other. It would be best to remove the old_price and discount features then
    since it has a lot of overlap with latest_price.
    Additionally, only one of ratings and reviews should be kept since they are heavily
    correlated with each other.
    '''

    # removing old_price, discount, star_rating and ratings columns
    laptop_data = laptop_data.drop(labels = ['old_price', 'discount', 'star_rating', 'ratings'], axis = 1)
    laptop_data.describe()

    # missing data
    total = laptop_data.isnull().sum().sort_values(ascending = False)
    percent = (laptop_data.isnull().sum()/laptop_data.isnull().count()).sort_values(ascending = False)
    missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    missing_data

    # check for odd values for cleaning
    laptop_data = laptop_data.convert_dtypes()
    column_names = laptop_data.columns.values.tolist()

    for feature in column_names:
        try:
            print(feature + ':')
            feature_uniques = laptop_data[feature].unique()
            feature_uniques = sorted(feature_uniques)
            print(feature_uniques)
            print()
        except:
            print('Probably has NA value')
            print()

    '''
    brand: 'Lenovo' and 'lenovo' are the only duplicates, just replace
    model: more inconsistency with model names here, like 'ThinkPad' vs 'Thinkpad', lower and str then see if more is needed
    processor_brand: '512', '64', 'First', and 'Pre-installed' kind of don't make sense, may remove
    processor_name: has NA values
    processor_gnrtn: almost half of the values are 'Missing', may just remove this column
    ram_gb: lots of weird values, should probably just impute with 8/16 since those are standard
    ram_type: has NA values
    ssd: all believable
    hdd: all believable
    os: 'Missing' should be checked
    os_bit: all believable
    graphic_card_gb: all believable
    weight: all believable
    display_size: some odd values here
    warranty: all believable
    Touchscreen: all believable
    msoffice: all believable
    latest_price: all believable
    reviews: all believable
    '''

    # check the features with NAN values
    total = laptop_data.isnull().sum().sort_values(ascending = False)
    # indeed the ram_type and processor_name columns have NAN values
    # have to check too which columns have entries like '0' or 'Missing'

    ### brand
    laptop_data.groupby('brand').brand.count()
    # only 3 'lenovo' entries vs 148 'Lenovo', replace 'lenovo' with 'Lenovo'
    laptop_data['brand'] = laptop_data['brand'].replace(to_replace = 'lenovo', value = 'Lenovo')

    ### model
    # some are just case sensitive so let's lowercase and strip whitespaces
    laptop_data['model'] = laptop_data['model'].str.lower()
    laptop_data['model'] = laptop_data['model'].str.strip()
    print(laptop_data['model'].unique())
    # some values are still close and might be the same, but cannot be sure since naming conventions are weird

    ### processor_brand
    laptop_data.groupby('processor_brand').processor_brand.count()
    # 512, 64, First, Pre-installed and M. 2 don't make sense
    # Probably outliers, should be safe to remove
    odd_brands = ['512', '64', 'First', 'Pre-installed', 'M.2']
    for off_brand in odd_brands:
        laptop_data = laptop_data[laptop_data['processor_brand'] != off_brand]

    ### processor_name
    # has 1 NA value
    # on checking, that entry is missing a lot of data, so should probably just drop it
    laptop_data = laptop_data.dropna(subset=['processor_name'])

    ### processor_gnrtn
    # almost half the column is 'Missing', should probably just remove this column
    laptop_data = laptop_data.drop(labels = 'processor_gnrtn', axis = 1)

    ### ram_gb
    # odd values and nans, but this is quite important so remove weird values then impute nans
    # remove weird ram values
    odd_rams = ['15.6', '5']
    for odd_ram in odd_rams:
        laptop_data = laptop_data[laptop_data['ram_gb'] != odd_ram]

    # impute ram values for remaining
    # imputer
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
    imputer = imputer.fit(laptop_data[['ram_gb']])
    laptop_data['ram_gb'] = imputer.transform(laptop_data[['ram_gb']])

    ### ram_type
    # DDR4 is most common by a large margin, so unscientifically replace nans with DDR4
    laptop_data['ram_type'] = laptop_data['ram_type'].replace(np.nan, 'DDR4')

    ### ssd
    ### hdd
    # both believable, just know that 0 means it doesnt have one (but it's odd if both are 0)
    # check that no entry exists where both ssd and hdd are 0
    print(len(laptop_data[(laptop_data['ssd'] == 0) & (laptop_data['hdd'] == 0)])) # result is 0

    ### os
    # all retail Apple laptops use MacOS, and it's doubtful that non-Windows machines are sold
    # at retail shops where the data is from.
    # check how many Missings there are with Apple laptops
    print(len(laptop_data[(laptop_data['os'] == 'Missing') & (laptop_data['brand'] == 'APPLE')])) # result is 0
    # replace the Missings with Windows
    laptop_data['os'] = laptop_data['os'].replace('Missing', 'Windows')

    ### os_bit, graphic_card_gb and weight are all good

    ### display_size
    # i am assuming these are inches, so '0', 'All', and 'ITW)' should be removed
    # it's hard to guess the display specs, so agreeable to remove the 'All' and 'ITW)' entries
    odd_displays = ['All', 'ITW)']
    for odd_display in odd_displays:
        laptop_data = laptop_data[laptop_data['display_size'] != odd_display]
    # since it's a integer value, mean might work
    laptop_data['display_size'] = laptop_data['display_size'].astype('float')
    display_imputer = SimpleImputer(missing_values = 0, strategy = 'mean')
    display_imputer = display_imputer.fit(laptop_data[['display_size']])
    laptop_data['display_size'] = display_imputer.transform(laptop_data[['display_size']])

    ### all the data should be cleaned by now

    '''
    At this point just make the datatypes correct
    '''

    print(laptop_data.dtypes)
    # ssd and graphic_card_gb can be made into ints
    laptop_data['ssd'] = laptop_data['ssd'].astype('int64')
    laptop_data['graphic_card_gb'] = laptop_data['graphic_card_gb'].astype('int64')

    print(laptop_data.dtypes)

    # make everything lowercase too for fun
    laptop_data = laptop_data.applymap(lambda s: s.lower() if type(s) == str else s)
    laptop_data.rename(columns={'Touchscreen':'touchscreen'}, inplace = True)

    # can write to csv now
    laptop_data.to_csv('final_laptop_data.csv', index = False)

    ###

    # redo some plots
    new_corrmat = laptop_data.corr()
    f, ax = plt.subplots(figsize = (12, 12))
    sns.heatmap(corrmat, vmax = 0.8, annot = True, square = True, xticklabels = 1, yticklabels = 1)

    '''
    plot shows ram_gb, ssd and graphic_card_gb have high correlation with latest_price,
    expected because these are all desirable features when shopping for a laptop
    funnily enough, hdd has negative correlation with latest_price most likely due
    to the shift from hdd to ssd, and hdd laptops are more often than not older and thus
    a worse value than newer entries into the market
    expectedly, higher specs are more closely bundled with each other too, so there is a
    high correlation between ssd and graphic_card_gb
    if the processor_brand and processor_name were one hot encoded, it would be expected
    that later generation cpus come bundled with higher specs as well and would also bump up
    the price of the machines
    otherwise, os_bit, display_size, and warranty don't seem to affect pricing that much
    '''

    high_correlation = ['ram_gb', 'ssd', 'graphic_card_gb']
    for feature in high_correlation:
        title = str(feature) + ' vs. latest_price'
        laptop_data.plot.scatter(x = feature, y = 'latest_price', ylim = (0, 350000))
        plt.title(title)
        box_data = pd.concat([laptop_data['latest_price'], laptop_data[feature]], axis = 1)
        f, ax = plt.subplots(figsize = (8, 6))
        fig = sns.boxplot(x = feature, y = 'latest_price', data = box_data)
        fig.set_title(title)
        fig.axis(ymin = 0, ymax = 350000)

    # for brands
    feature = 'brand'
    data = pd.concat([laptop_data['latest_price'], laptop_data[feature]], axis = 1)
    f, ax = plt.subplots(figsize = (8, 6))
    fig = sns.boxplot(x = feature, y = 'latest_price', data = laptop_data)
    plt.xticks(rotation=90)
    fig.set_title('brand vs. latest_price')
    fig.axis(ymin = 0, ymax = 350000)

    