def func(dataset):
    import pandas as pd
    import streamlit as st
    import numpy as np
    import warnings
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    #dataset = pd.read_csv('Wine_Quality.csv')
    warnings.filterwarnings("ignore")

    sns.set(style="white", color_codes= True)

    dataset['wine type'].value_counts()

    plt.figure(figsize=(10,10))
    sns.heatmap(dataset.corr(),cbar=True,annot=True,cmap='Blues')

    train, test = train_test_split(dataset, test_size=0.2)

    train_dfc = train.copy(deep=True)
    train_dfr = train.copy(deep=True)
    test_dfc = test.copy(deep=True)
    test_dfr = test.copy(deep=True)

    train_dfc.insert(13,'quality_level','')
    train_dfr.insert(13,'quality_level','')
    test_dfc.insert(13,'quality_level','')
    test_dfr.insert(13,'quality_level','')

    train_dfc['quality_level'] = np.where(train_dfc['quality'] > 5, 1, 0)
    train_dfr['quality_level'] = np.where(train_dfr['quality'] > 5, 1, 0)
    test_dfc['quality_level'] = np.where(test_dfc['quality'] > 5, 1, 0)
    test_dfr['quality_level'] = np.where(test_dfr['quality'] > 5, 1, 0)

    ax = sns.countplot(x="quality_level", hue="wine type", data=train_dfc)

    train_dfc.drop(['wine type'], axis=1, inplace=True)
    train_dfr.drop(['wine type'], axis=1, inplace=True)
    test_dfc.drop(['wine type'], axis=1, inplace=True)
    test_dfr.drop(['wine type'], axis=1, inplace=True)

    print(train_dfc.head())

    target = train_dfr['quality'].copy()
    features = train_dfr.drop(['quality'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 101)
    print(y_test)

    from sklearn import linear_model

    regression_model= linear_model.LinearRegression(normalize=True)
    regression_model.fit(X_train,y_train)

    coeff = regression_model.coef_
    intercept = regression_model.intercept_
    print('coeff =', coeff)
    print('intercept =', intercept)

    predicted_quality = pd.DataFrame(regression_model.predict(X_test), columns =['Predicted Quality'])
    actual_quality = y_test.reset_index(drop=True)

    df_actual_vs_predicted = pd.concat([actual_quality, predicted_quality], axis=1)
    df_actual_vs_predicted.T

    predictions = regression_model.predict(X_test)

    from sklearn.metrics import r2_score
    print("R-squared value=",r2_score(y_test, predictions))

    return df_actual_vs_predicted