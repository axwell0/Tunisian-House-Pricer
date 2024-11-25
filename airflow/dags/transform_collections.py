import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def convert_price_to_tnd(df, price_column='price', eur_to_tnd=3.34):
    h = df.copy()
    h[price_column] = (
            h[price_column]
            .str.extract(r'(\d[\d\s]*)')[0]
            .str.replace(r'\s+', '', regex=True).str.replace(',', '', regex=True)
            .astype('Int64')
            * h[price_column].apply(lambda x: eur_to_tnd if 'EUR' in x else 1)
    )
    h = h.reset_index(drop=True)
    return h.copy()


def transform_affare(affare_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply specific processing for the 'Affare' collection.
    """
    affare_df = affare_df[affare_df['price'].isna() == False]
    affare_df = convert_price_to_tnd(affare_df, 'price')
    affare_df.drop(['Meublée', 'posting_date', 'Adresse'], axis=1, inplace=True)
    affare_df['Chambre'] = affare_df['Chambre'].str.extract('(\d+)').astype('Int64')
    affare_df['Salles de bains'] = affare_df['Salles de bains'].str.extract('(\d+)').astype('Int64')
    affare_df['Superficie'] = affare_df['Superficie'].str.extract('(\d+)').astype('Int64')
    affare_df['city'] = affare_df['location'].str.split(' - ', expand=True).loc[:, 1].str.lower()
    affare_df['state'] = affare_df['location'].str.split(' - ', expand=True).loc[:, 0].str.lower()
    affare_df['Type'] = affare_df['Type'].fillna('villa')
    affare_df.rename(
        {"Chambre": 'n_bedrooms',
         'Salles de bains': 'n_bathrooms',
         'Superficie': 'area'}
        , axis='columns', inplace=True)
    affare_df['city'] = affare_df['city'].apply(lambda x: 'hammamet' if 'hammamet' in x else x)
    affare_df = affare_df[(affare_df['area'].isna() == False) & (affare_df['n_bathrooms'].isna() == False)]

    return affare_df


def transform_menzili(menzili_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply specific processing for 'menzili' collection.
    """
    menzili_df['price'] = menzili_df['price'].str.replace(' ', '').str.extract('(\d+)').astype('Int64')
    menzili_df.dropna(subset=['Surf terrain', 'Salle de bain', 'Chambres', 'price', 'location'], inplace=True)
    menzili_df.drop(['Piéces Totale', 'Année construction', 'Surf habitable', 'misc'], axis=1, inplace=True)
    menzili_df.rename({'Chambres': 'n_bedrooms', 'Salle de bain': 'n_bathrooms', 'Surf terrain': 'area'}, axis='columns',
                      inplace=True)
    menzili_df['n_bedrooms'] = menzili_df['n_bedrooms'].str.replace('+', '').astype('Int64')
    menzili_df['n_bathrooms'] = menzili_df['n_bathrooms'].str.replace('+', '').astype('Int64')
    menzili_df['area'] = menzili_df['area'].str.extract('(\d+)').astype('Int64')
    menzili_df['state'] = menzili_df['location'].str.split(', ', expand=True).loc[:, 2].str.replace('é', 'e').str.lower()
    menzili_df['city'] = menzili_df['location'].str.split(', ', expand=True).loc[:, 1].str.replace('é', 'e').str.lower()
    menzili_df.dropna(subset=['city', 'state'], inplace=True)
    menzili_df['city'] = menzili_df['city'].str.replace('djerba - midoun', 'djerba').apply(
        lambda x: 'hammamet' if 'hammamet' in x else x)

    return menzili_df




def transform_aggregate(affare_df : pd.DataFrame, menzili_df : pd.DataFrame) -> pd.DataFrame:
    aggregate_df = pd.concat([affare_df, menzili_df])
    aggregate_df = aggregate_df[(aggregate_df['price'] > 80000) & (aggregate_df['price'] <= 1000000)]
    aggregate_df = aggregate_df[(aggregate_df['n_bedrooms'] <= 7) & (aggregate_df['n_bedrooms'] >= 1)]
    aggregate_df = aggregate_df[(aggregate_df['n_bathrooms'] >= 1) & (aggregate_df['n_bathrooms'] < 7)]
    aggregate_df = aggregate_df[(aggregate_df['area'] >= 100) & (aggregate_df['area'] <= 1500)]

    label_encoder = LabelEncoder()
    aggregate_df['type_encoded'] = label_encoder.fit_transform(aggregate_df['Type'])
    aggregate_df['price_log'] = np.log1p(aggregate_df['price'])
    aggregate_df['area_log'] = np.log1p(aggregate_df['area'])
    return aggregate_df