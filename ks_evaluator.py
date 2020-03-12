from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from typing import List, Tuple, Optional, Union

import pyspark.sql.functions as F


class KolmogorovSmirnovEvaluator:

    NO_SAMPLES = 2

    COUNT = 'count'
    CUMSUM = '__cumsum__'
    EMPIRICAL_CDF = '__ecdf__'

    EMPIRICAL_CDF_NEG = '__ecdf_0__'
    EMPIRICAL_CDF_POS = '__ecdf_1__'
    DISTANCE = '__distance__'
    PARTITION = '__partition__'

    def __init__(self, probability_col: str, actual_label_col: str,
                 positive_label: Optional[Union[int, float, str]] = None,
                 negative_label: Optional[Union[int, float, str]] = None,
                 probability_partitions: Optional[List[float]] = None):

        self.probability_col = probability_col
        self.actual_label_col = actual_label_col
        self.p = positive_label
        self.n = negative_label
        self.partitions = probability_partitions

        assert isinstance(self.p, type(self.n)), "Labels must be of the same type."

        if self.partitions is not None:
            self.partitions = sorted(self.partitions, reverse=True)
            assert self.partitions[0] < 1 and self.partitions[-1] > 0, "Probabilities must be in the range 0 < p < 1."

    def _set_labels_from_count_aggregate(self, count_df: DataFrame):
        self.p, self.n = [row[self.actual_label_col] for row in count_df.collect()]

    def _get_cdfs(self, prob_df: DataFrame, count_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        window_cdf = Window.partitionBy(self.actual_label_col).orderBy(self.probability_col)
        df_cumsum = prob_df.withColumn(self.CUMSUM, F.count(self.probability_col).over(window_cdf))

        df_cdf = df_cumsum.join(count_df, on=self.actual_label_col, how='left') \
            .withColumn(self.EMPIRICAL_CDF, F.col(self.CUMSUM) / F.col(self.COUNT))

        df_cdf_0 = df_cdf.where(F.col(self.actual_label_col) == self.n) \
            .select(F.col(self.probability_col), F.col(self.EMPIRICAL_CDF).alias(self.EMPIRICAL_CDF_NEG))
        df_cdf_1 = df_cdf.where(F.col(self.actual_label_col) == self.p) \
            .select(F.col(self.probability_col), F.col(self.EMPIRICAL_CDF).alias(self.EMPIRICAL_CDF_POS))
        return df_cdf_0, df_cdf_1

    def _get_distances(self, prob_df: DataFrame, df_cdf_0: DataFrame, df_cdf_1: DataFrame) -> DataFrame:
        window_fill = Window.orderBy(self.probability_col).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        df_ks = prob_df.select(self.probability_col) \
            .join(df_cdf_0, on=self.probability_col, how='left') \
            .join(df_cdf_1, on=self.probability_col, how='left') \
            .withColumn(self.EMPIRICAL_CDF_NEG, F.last(self.EMPIRICAL_CDF_NEG, ignorenulls=True).over(window_fill)) \
            .withColumn(self.EMPIRICAL_CDF_POS, F.last(self.EMPIRICAL_CDF_POS, ignorenulls=True).over(window_fill)) \
            .fillna(0) \
            .withColumn(self.DISTANCE, F.abs(F.col(self.EMPIRICAL_CDF_NEG) - F.col(self.EMPIRICAL_CDF_POS)))
        return df_ks

    def _get_ks_statistic(self, df_distances: DataFrame) -> dict:
        if not self.partitions:
            ks_stat_row = df_distances.agg(F.max(self.DISTANCE).alias(self.DISTANCE)).collect()
            ks_stat = ks_stat_row[0][self.DISTANCE]

            result = {
                'statistic': ks_stat,
                'ks_table': [
                    {'upper_bound': 1,
                     'lower_bound': 0,
                     'statistic': ks_stat}
                ]
            }
        else:
            df_ks_partitioned = df_distances.withColumn(self.PARTITION, F.lit(0))

            for idx, threshold in enumerate(self.partitions):
                df_ks_partitioned = df_ks_partitioned\
                    .withColumn(self.PARTITION,
                                F.when(F.col(self.probability_col) <= threshold, F.lit(idx+1))
                                .otherwise(F.col(self.PARTITION)))
            ks_stat_partitioned = df_ks_partitioned.groupBy(self.PARTITION)\
                .agg(F.max(self.DISTANCE).alias(self.DISTANCE))

            ks_stat_row = ks_stat_partitioned.agg(F.max(self.DISTANCE).alias(self.DISTANCE)).collect()
            ks_stat = ks_stat_row[0][self.DISTANCE]

            bounded_partition = [1.0] + self.partitions + [0.0]
            ks_stat_rows = ks_stat_partitioned.collect()
            ks_table = [
                {
                    'upper_bound': bounded_partition[row[self.PARTITION]],
                    'lower_bound': bounded_partition[1+row[self.PARTITION]],
                    'statistic': row[self.DISTANCE]
                }
                for row in ks_stat_rows
            ]

            result = {
                'statistic': ks_stat,
                'ks_table': ks_table

            }

        return result

    def evaluate(self, df: DataFrame):

        prob_df = df.select(self.probability_col, self.actual_label_col)
        df_counts = prob_df.groupBy(self.actual_label_col).count()

        assert len(df_counts.take(1+self.NO_SAMPLES)) == self.NO_SAMPLES, 'Kolmogorov-Smirnov evaluator is for ' \
                                                                          'binary classification only.'

        if not self.p or not self.n:
            self._set_labels_from_count_aggregate(df_counts)

        df_cdf_0, df_cdf_1 = self._get_cdfs(prob_df, df_counts)
        df_distances = self._get_distances(prob_df, df_cdf_0, df_cdf_1)
        result = self._get_ks_statistic(df_distances)

        return result
