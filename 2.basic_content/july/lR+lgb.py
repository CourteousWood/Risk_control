from sklearn.preprocessing import OneHotEncoder,LabelEncoder

lgb_train = lgb.Dataset(df_train[NUMERIC_COLS], df_train['bad_ind'], free_raw_data=False)
params = {
    'num_boost_round': 50,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 2,
    'metric': 'auc',
    'max_depth':1,
    'feature_fraction':1,
    'bagging_fraction':1,
}
model = lgb.train(params,lgb_train)
leaf = model.predict(df_train[NUMERIC_COLS],pred_leaf=True)
lgb_enc = OneHotEncoder()
lgb_enc.fit(leaf)
data_leaf = np.hstack((lgb_enc.transform(leaf).toarray(),df_train[NUMERIC_COLS]))


train, val, train_y, val_y = train_test_split(data_leaf,df_train['bad_ind'],test_size=0.2, random_state=random.choice(range(10000)))

lgb_lm = LogisticRegression(penalty='l1',C = 0.3,solver='liblinear')
lgb_lm.fit(train, train_y)

y_pred_lgb_lm_train = lgb_lm.predict_proba(train)[:, 1]
fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(train_y, y_pred_lgb_lm_train)

y_pred_lgb_lm = lgb_lm.predict_proba(val)[:, 1]
fpr_lgb_lm, tpr_lgb_lm, _ = roc_curve(val_y, y_pred_lgb_lm)