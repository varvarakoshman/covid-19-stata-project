import pandas as pd
import math


def join_data():
    # brent & gaz
    df_brent = pd.read_csv('data_to_join/brent-daily_csv.csv')
    df_gaz = pd.read_csv('data_to_join/gaz_daily.csv')
    df_b_g = pd.merge(left=df_brent, right=df_gaz, left_on='Date', right_on='Date', suffixes=('_brent', '_gaz'))

    # vti index
    df_wti = pd.read_csv('data_to_join/wti-daily_csv.csv')
    df_b_g_w = pd.merge(left=df_b_g, right=df_wti, left_on='Date', right_on='Date')
    df_b_g_w.rename(columns={"Price": "Price_wti"}, inplace=True)

    # vix index
    df_vix = pd.read_csv('data_to_join/vix-daily_csv.csv')
    df_b_g_w_v = pd.merge(left=df_b_g_w, right=df_vix, left_on='Date', right_on='Date')

    # currency
    df_currency = pd.read_csv('data_to_join/currency_daily_csv.csv')
    currency_countries_groups = df_currency.groupby(df_currency.Country)
    all_countries = currency_countries_groups.groups.keys()
    df_b_g_w_v_c = df_b_g_w_v
    for country in all_countries:
        country_df = currency_countries_groups.get_group(country)
        df_b_g_w_v_c = pd.merge(left=df_b_g_w_v_c, right=country_df, left_on='Date', right_on='Date')
        new_name = '%s_Exchange_rate' % country
        df_b_g_w_v_c.rename(columns={"Exchange rate": new_name}, inplace=True)
        df_b_g_w_v_c.drop(columns=['Country'], inplace=True)

    # google mobility
    df_mobility = pd.read_csv('data_to_join/Global_Mobility_Report 3.csv')
    df_mobility_us = df_mobility[df_mobility.country_region_code == 'US']

    # fill in missing values with prev value

    # prepare needed dates
    df_google_mobility_groups = df_mobility_us.groupby(df_mobility_us.sub_region_1)
    random_group = df_google_mobility_groups.get_group('Vermont')
    df_random_group_unique_dates = random_group.groupby(random_group.date).mean().reset_index()
    df_dates_unique = df_random_group_unique_dates[['date']]
    df_dates_unique.rename(columns={"date": 'Date'}, inplace=True)

    # join currencies on these dates
    currencies_everyday = pd.merge(left=df_dates_unique, right=df_b_g_w_v_c, on='Date', how='left')
    # remove nans from start and end
    currencies_everyday_filtered = currencies_everyday[3:91]

    prev = currencies_everyday_filtered[:].iloc[0]
    column_names = list(map(lambda x: x[0], prev.items()))
    column_names.remove('Date')
    for row_index, row in currencies_everyday_filtered.iterrows():
        if math.isnan(float(row['Price_brent'])):
            for column_name in column_names:
                currencies_everyday_filtered.loc[row_index, column_name] = prev[column_name]
        else:
            prev = row

    df_mobility_us.drop(columns=['country_region_code', 'country_region'], inplace=True)
    df_google_mobility_groups = df_mobility_us.groupby(df_mobility_us.sub_region_1)
    for state_group in df_google_mobility_groups:
        state_name = state_group[0]
        state_df_city_mean = state_group[1].groupby(state_group[1].date).mean()
        state_df_city_mean['Date'] = state_df_city_mean.index
        state_df_city_mean = state_df_city_mean.reset_index(drop=True)
        state_df_city_mean.rename(
            columns={'retail_and_recreation_percent_change_from_baseline': 'retail_and_recreation_%s_avg' % state_name,
                     'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_and_pharmacy_%s_avg' % state_name,
                     'parks_percent_change_from_baseline': 'parks_%s_avg' % state_name,
                     'transit_stations_percent_change_from_baseline': 'transit_stations_%s_avg' % state_name,
                     'workplaces_percent_change_from_baseline': 'workplaces_%s_avg' % state_name,
                     'residential_percent_change_from_baseline': 'residential_%s_avg' % state_name}, inplace=True)
        currencies_everyday_filtered = pd.merge(left=currencies_everyday_filtered, right=state_df_city_mean, on='Date', how='left')

    # apple mobility
    df_apple_mobility = pd.read_csv('data_to_join/apple_mobility.csv')
    df_apple_mobility_filtered = df_apple_mobility[df_apple_mobility.country == 'United States']
    df_apple_mobility_filtered.drop(columns=['geo_type', 'wikidata', 'lat', 'lng', 'short_code', 'iso_3166_alpha_2',
                                             'iso_3166_alpha_3', 'iso_3166_numeric', 'alternative_name', 'country'],
                                    inplace=True)
    df_apple_mobility_filtered.rename(columns={"sub-region": 'subregion'}, inplace=True)
    df_apple_mobility_groups = df_apple_mobility_filtered.groupby(df_apple_mobility_filtered.subregion)
    for state_group in df_apple_mobility_groups:
        state_name = state_group[0]
        state_df = state_group[1].groupby(df_apple_mobility_filtered.transportation_type).mean().T
        state_df['Date'] = state_df.index
        state_df = state_df.reset_index(drop=True)
        state_df.rename(columns={"driving": 'driving_%s_avg' % state_name,
                                 "transit": 'transit_%s_avg' % state_name,
                                 "walking": 'walking_%s_avg' % state_name}, inplace=True)
        currencies_everyday_filtered = pd.merge(left=currencies_everyday_filtered, right=state_df, on='Date', how='left')

    # covid cases and deaths US
    df_covid_cases = pd.read_csv('data_to_join/covid_us_county.csv')
    df_covid_cases = df_covid_cases[['state', 'date', 'cases', 'deaths']].copy()
    covid_cases_state_groups = df_covid_cases.groupby(df_covid_cases.state)
    for state_group in covid_cases_state_groups:
        state_name = state_group[0]
        state_df_city_mean = state_group[1].groupby(state_group[1].date).mean()
        state_df_city_mean['Date'] = state_df_city_mean.index
        state_df_city_mean = state_df_city_mean.reset_index(drop=True)
        state_df_city_mean.rename(columns={"cases": '%s_cases' % state_name,
                                           "deaths": '%s_deaths' % state_name}, inplace=True)
        currencies_everyday_filtered = pd.merge(left=currencies_everyday_filtered, right=state_df_city_mean, on='Date', how='left')

    result_file_name = 'gaz_brent_vix_wti_currency_mobility_covidStata_merged.csv'
    currencies_everyday_filtered.to_csv(result_file_name, sep=',')


if __name__ == '__main__':
    join_data()
