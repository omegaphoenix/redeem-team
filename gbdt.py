import time
import xgboost as xgb

def format_blend_data(ratings, models, output):
    out = open(output, 'w')

    ms = []
    for model in models:
        ms.append(open(model, 'r'))

    if (ratings == ""):
        ratings = ["0"] * 2749898 # number of points in qual
    else:
        f = open(ratings, 'r')
        ratings = f.read().splitlines()
        f.close()

    i = 0
    for rating in ratings:
        rating = rating.strip()
        out.write(rating + " ")
        feature_id = 0
        for m in ms:
            value = m.readline().strip()
            out.write(str(feature_id) + ":" + value + " ")
            feature_id += 1
        out.write("\n")
        i += 1
    print "number of ratings: " + str(i)

    out.close()
    for m in ms:
        m.close()

def save_output(arr, ofile):
    with open(ofile, 'w') as f:
        for prediction in arr:
            f.write(str(prediction) + "\n")

def main():
    start = time.time()

    train_file = 'gbdt_train.txt'
    test_file = 'gbdt_test.txt'

    # models = ["blend/tsvdpp_220factors_30bins_38epochs_4dta.out",
    # "blend/pure_rbm_v3_factors_200_epoch_44_T_9_4dta.txt",
    # "blend/dates_4dta.txt"]
    # format_blend_data("blend/predictions_4dta.txt", models, train_file)

    # models = ["blend/tsvdpp_220factors_30bins_38epochs_5dta.out",
    # "blend/pure_rbm_v3_factors_200_epoch_44_T_9_5dta.txt",
    # "blend/dates_5dta.txt"]
    # format_blend_data("", models, test_file)

    t1 = time.time()
    print "Writing input files took: " + str(t1 - start) + " s"

    dtrain = xgb.DMatrix(train_file)
    dtest = xgb.DMatrix(test_file)
    param = {
        'max_depth':4,
        'eta':0.2, # shrinkage / learning rate
        'subsample':1.0,
        # 'lambda':0.1,
        'silent':1,
        'objective':'reg:linear',
        'eval_metric':'rmse',
        'seed':0}
    num_round = 40 # N_boost / number of trees
    bst = xgb.train(param, dtrain, num_round)

    preds = bst.predict(dtest)
    print preds
    save_output(preds, "gbdt_blend.out");

    end = time.time()
    print "GBDT took: " + str(end - start) + " s"

if __name__ == "__main__":
	main()
