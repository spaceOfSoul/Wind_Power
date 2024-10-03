input_path = "../input/"
model_output = "../models/"
test_size = 0.2
xgb_params = {
    'tree_method': 'auto',
    # 'n_estimators': 1000,
    # 'max_depth': 8,
    'n_jobs': -1,
    'random_state': 1,
    'device': 'cuda',
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae'
}