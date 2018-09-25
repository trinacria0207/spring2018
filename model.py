import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from random import randint
import pandas as pd
from sklearn.model_selection import train_test_split

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(1103, input_dim=1103, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(625, input_dim=384, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def larger_model():
    model = Sequential()
    model.add(Dense(50, input_dim=25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

def main():
    seed = 2048
    np.random.seed(seed)
    data = pd.read_csv("clean_2.csv")
    X_train  = data.drop(['loss'],axis =1)
    y_train  = data['loss'] # [0 if i == 0 else 1 for i in data['loss']]
    test_data = pd.read_csv("test_2.csv")
    X_test = test_data.drop(['id'],axis=1)
    id_list = list(test_data['id'])
    T = X_train
    T['loss'] = y_train
    T = T.loc[T['loss'] > 0]
    X_train_reg = T.drop(['loss'],axis = 1)
    y_train_reg = T['loss']
    X_train_cla = X_train.drop(['loss'],axis =1)
    y_train_cla = [0 if i == 0 else 1 for i in y_train]
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=15, batch_size=50)))
    estimators.append(('mlp', GradientBoostingClassifier(n_estimators=160, max_features='auto', max_depth = 5, verbose=1)))
    pipeline = Pipeline(estimators)
    # kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    # results = cross_val_score(pipeline, X_train_cla, y_train_cla, cv=kfold)
    # print("Loss: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    pipeline.fit(X_train_cla, y_train_cla)
    train_cla_pred = pipeline.predict(X_train_cla)
    estimators2 = []
    estimators2.append(('standardize', StandardScaler()))
    estimators2.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=10, batch_size=50)))
    pipeline2 = Pipeline(estimators2)
    # kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    # results = cross_val_score(pipeline2, X_train_reg, y_train_reg, cv=kfold)
    # print("Loss: %.2f%% (%.2f%%)" % (-results.mean()*100, results.std()*100))

    
   
    pipeline2.fit(X_train_reg, y_train_reg)
    print(X_test.shape)
    y_test_cla = pipeline.predict(X_test)
    y_test_reg = pipeline2.predict(X_test)
    y_test = []
    for ind in range(len(y_test_cla)):
        if y_test_cla[ind] == 0:
            y_test.append(0)
        else:
            y_test.append(int(y_test_reg[ind]+0.5))

    d = {'id': id_list , 'loss' : y_test }
    ans = pd.DataFrame(d)
    ans.to_csv("sub_m.csv", sep=',',  index=False)
    

if __name__ == '__main__':
    main()
