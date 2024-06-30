from cross_validation import *
from prepare_data import *


def main():
    args, _ = config.set_config()
    sub_to_run = np.arange(args.subjects)
    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=args.fold, reproduce=args.reproduce)


if __name__ == '__main__':
    for i in range(5):
        main()
