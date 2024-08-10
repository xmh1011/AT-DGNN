from config.config import *
from train.cross_validation import *
from train.prepare_data import *

if __name__ == '__main__':
    args, _ = set_config()
    sub_to_run = np.arange(args.subjects)
    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=args.fold, reproduce=args.reproduce)
