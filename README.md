# ks-evaluator
2-sample Kolmogorov-Smirnov distance calculator for evaluating binary classification model predictions.

## What is this?
This script computes the [Kolmogorov-Smirnov distance](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test) between two empirical distributions. In this case, the empirical distributions are the distributions of the probability of being in the positive class (usually denoted class `1`) between samples that are actually in the positive class and samples that are actually in the negative class.

The Kolmogorov-Smirnov distance gives an idea of how close the two empirical distributions are: the further apart they are, the better.

Currently, this script does not perform a test of significance for the Kolmogorov-Smirnov score. It will be added if absolutely necessary.

## How do you use this?
Simply copy and paste the `KolmogorovSmirnovEvaluator` class or the entire `ks_evaluator.py` into your project, and import the `KolmogorovSmirnovEvaluator` class. The Evaluator's constructor takes 6 arguments:
1. `df`: A Spark DataFrame.
1. `probability_col`: Name of the probability column.
1. `actual_label_col`: Name of the column which contains the *correct* label - *NOT* the prediction!
1. `positive_label`: (Optional) The positive label.
1. `negative_label`: (Optional) The negative label. Must be of the same type as the `positive_label`. If either of the `positive_label` or the `negative_label` is left empty, the evaluator will try to infer them from the data.
1. `probability_partitions`: (Optional) A list of partitions for which the Kolmogorov-Smirnov distance will be calculated. For example, if the `probability_partitions = [0.2, 0.4]`, then the script will compute the Kolmogorov-Smirnov distance in the following intervals: `[0, 0.2]`, `(0.2, 0.4]`, and `(0.4, 1]`. The elements of this list *must* be in the range `(0, 1)`.

After constructing the Evaluator object, you can perform the evaluation using `evaluator.evaluate()`.

## What does it output?
The output format goes like this:
```
{'ks_table': [{'lower_bound': 0.0,
               'statistic': 0.061130753217147844,
               'upper_bound': 0.2},
              {'lower_bound': 0.2,
               'statistic': 0.19390728562772036,
               'upper_bound': 0.4},
              {'lower_bound': 0.4,
               'statistic': 0.26037290640448074,
               'upper_bound': 0.6},
              {'lower_bound': 0.6,
               'statistic': 0.2647914886645406,
               'upper_bound': 0.8},
              {'lower_bound': 0.8,
               'statistic': 0.1189296756699938,
               'upper_bound': 1.0}],
 'statistic': 0.2647914886645406}
```
where:
1. `ks_table` is the collection of Kolmogorov-Smirnov distances for the given partitions; if no partitions were given at the time of construction, it will simply give the Kolmogorov-Smirnov distance for the entire `[0,1]` interval, i.e.
```
{'lower_bound': 0.0,
 'statistic': 0.2647914886645406,
 'upper_bound': 1.0}
```
2. `statistic` is just the Kolmogorov-Smirnov distance of the two empirical distributions.

## That's it!
If there are any issues, do submit them. Happy KS-ing!
