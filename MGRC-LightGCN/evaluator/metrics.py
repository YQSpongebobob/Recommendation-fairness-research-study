from logging import getLogger
import numpy as np
from collections import Counter
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TopK Metrics

class Hit(TopkMetric):

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result('hit', result)
        return metric_dict

    def metric_info(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int)


class MRR(TopkMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result('mrr', result)
        return metric_dict

    def metric_info(self, pos_index):
        idxs = pos_index.argmax(axis=1)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, idx in enumerate(idxs):
            if pos_index[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class MAP(TopkMetric):

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('map', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
        sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
        result = np.zeros_like(pos_index, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, pos_index.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


class Recall(TopkMetric):
    r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
       \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}

    :math:`|R(u)|` represents the item count of :math:`R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('recall', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('ndcg', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result


class Precision(TopkMetric):
    r"""Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
    out of all the recommended items. We average the metric for each user :math:`u` get the final result.

    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}

    :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result('precision', result)
        return metric_dict

    def metric_info(self, pos_index):
        return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)


# CTR Metrics


class GAUC(AbstractMetric):
    r"""GAUC (also known as Grouped Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve grouped by user. We weighted the index of each user :math:`u` by the number of positive
    samples of users to get the final result.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3219819.3219823>`__

    Note:
        It calculates the AUC score of each user, and finally obtains GAUC by weighting the user AUC.
        It is also not limited to k. Due to our padding for `scores_tensor` with `-np.inf`, the padding
        value will influence the ranks of origin items. Therefore, we use descending sort here and make
        an identity transformation  to the formula of `AUC`, which is shown in `auc_` function.
        For readability, we didn't do simplification in the code.

    .. math::
        \begin{align*}
            \mathrm {AUC(u)} &= \frac {{{|R(u)|} \times {(n+1)} - \frac{|R(u)| \times (|R(u)|+1)}{2}} -
            \sum\limits_{i=1}^{|R(u)|} rank_{i}} {{|R(u)|} \times {(n - |R(u)|)}} \\
            \mathrm{GAUC} &= \frac{1}{\sum_{u \in U} |R(u)|}\sum_{u \in U} |R(u)| \cdot(\mathrm {AUC(u)})
        \end{align*}

    :math:`rank_i` is the descending rank of the i-th items in :math:`R(u)`.
    """
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.meanrank']

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        mean_rank = dataobject.get('rec.meanrank').numpy()
        pos_rank_sum, user_len_list, pos_len_list = np.split(mean_rank, 3, axis=1)
        user_len_list, pos_len_list = user_len_list.squeeze(-1), pos_len_list.squeeze(-1)
        result = self.metric_info(pos_rank_sum, user_len_list, pos_len_list)
        return {'gauc': round(result, self.decimal_place)}

    def metric_info(self, pos_rank_sum, user_len_list, pos_len_list):
        """Get the value of GAUC metric.

        Args:
            pos_rank_sum (numpy.ndarray): sum of descending rankings for positive items of each users.
            user_len_list (numpy.ndarray): the number of predicted items for users.
            pos_len_list (numpy.ndarray): the number of positive items for users.

        Returns:
            float: The value of the GAUC.
        """
        neg_len_list = user_len_list - pos_len_list
        # check positive and negative samples
        any_without_pos = np.any(pos_len_list == 0)
        any_without_neg = np.any(neg_len_list == 0)
        non_zero_idx = np.full(len(user_len_list), True, dtype=np.bool)
        if any_without_pos:
            logger = getLogger()
            logger.warning(
                "No positive samples in some users, "
                "true positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= (pos_len_list != 0)
        if any_without_neg:
            logger = getLogger()
            logger.warning(
                "No negative samples in some users, "
                "false positive value should be meaningless, "
                "these users have been removed from GAUC calculation"
            )
            non_zero_idx *= (neg_len_list != 0)
        if any_without_pos or any_without_neg:
            item_list = user_len_list, neg_len_list, pos_len_list, pos_rank_sum
            user_len_list, neg_len_list, pos_len_list, pos_rank_sum = map(lambda x: x[non_zero_idx], item_list)

        pair_num = (user_len_list + 1) * pos_len_list - pos_len_list * (pos_len_list + 1) / 2 - np.squeeze(pos_rank_sum)
        user_auc = pair_num / (neg_len_list * pos_len_list)
        result = (user_auc * pos_len_list).sum() / pos_len_list.sum()
        return result


class AUC(LossMetric):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
    the area under the ROC curve.

    .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users. We call the interface
        in `scikit-learn`, and code calculates the metric using the variation of following formula.

    .. math::
        \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
        \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

    :math:`M` denotes the number of positive items.
    :math:`N` denotes the total number of user-item interactions.
    :math:`rank_i` denotes the descending rank of the i-th positive item.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('auc', dataobject)

    def metric_info(self, preds, trues):
        fps, tps = _binary_clf_curve(trues, preds)
        if len(fps) > 2:
            optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        if fps[-1] <= 0:
            logger = getLogger()
            logger.warning("No negative samples in y_true, " "false positive value should be meaningless")
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            logger = getLogger()
            logger.warning("No positive samples in y_true, " "true positive value should be meaningless")
            tpr = np.repeat(np.nan, tps.shape)
        else:
            tpr = tps / tps[-1]

        result = sk_auc(fpr, tpr)
        return result


# Loss-based Metrics


class MAE(LossMetric):
    r"""MAE_ (also known as Mean Absolute Error regression loss) is used to evaluate the difference between
    the score predicted by the model and the actual behavior of the user.

    .. _MAE: https://en.wikipedia.org/wiki/Mean_absolute_error

    .. math::
        \mathrm{MAE}=\frac{1}{|{S}|} \sum_{(u, i) \in {S}}\left|\hat{r}_{u i}-r_{u i}\right|

    :math:`|S|` represents the number of pairs in :math:`S`.
    """
    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('mae', dataobject)

    def metric_info(self, preds, trues):
        return mean_absolute_error(trues, preds)


class RMSE(LossMetric):
    r"""RMSE_ (also known as Root Mean Squared Error) is another error metric like `MAE`.

    .. _RMSE: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    .. math::
       \mathrm{RMSE} = \sqrt{\frac{1}{|{S}|} \sum_{(u, i) \in {S}}(\hat{r}_{u i}-r_{u i})^{2}}
    """
    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('rmse', dataobject)

    def metric_info(self, preds, trues):
        return np.sqrt(mean_squared_error(trues, preds))


class LogLoss(LossMetric):
    r"""Logloss_ (also known as logistic loss or cross-entropy loss) is used to evaluate the probabilistic
    output of the two-class classifier.

    .. _Logloss: http://wiki.fast.ai/index.php/Log_Loss

    .. math::
        LogLoss = \frac{1}{|S|} \sum_{(u,i) \in S}(-((r_{u i} \ \log{\hat{r}_{u i}}) + {(1 - r_{u i})}\ \log{(1 - \hat{r}_{u i})}))
    """
    smaller = True

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('logloss', dataobject)

    def metric_info(self, preds, trues):
        eps = 1e-15
        preds = np.float64(preds)
        preds = np.clip(preds, eps, 1 - eps)
        loss = np.sum(-trues * np.log(preds) - (1 - trues) * np.log(1 - preds))
        return loss / len(preds)


class ItemCoverage(AbstractMetric):
    r"""ItemCoverage_ computes the coverage of recommended items over all items.

    .. _ItemCoverage: https://en.wikipedia.org/wiki/Coverage_(information_systems)

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/1864708.1864761>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

    .. math::
       \mathrm{Coverage@K}=\frac{\left| \bigcup_{u \in U} \hat{R}(u) \right|}{|I|}
    """
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items')
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = '{}@{}'.format('itemcoverage', k)
            metric_dict[key] = round(self.get_coverage(item_matrix[:, :k], num_items), self.decimal_place)
        return metric_dict

    def get_coverage(self, item_matrix, num_items):
        """Get the coverage of recommended items over all items

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            num_items(int): the total number of items.

        Returns:
            float: the `coverage` metric.
        """
        unique_count = np.unique(item_matrix).shape[0]
        return unique_count / num_items


class AveragePopularity(AbstractMetric):
    r"""AveragePopularity computes the average popularity of recommended items.

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__.

    .. math::
        \mathrm{AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U } \frac{\sum_{i \in R_{u}} \phi(i)}{|R_{u}|}

    :math:`\phi(i)` is the number of interaction of item i in training data.
    """
    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ['rec.items', 'data.count_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and the popularity of items in training data"""
        item_counter = dataobject.get('data.count_items')
        item_matrix = dataobject.get('rec.items')
        return item_matrix.numpy(), dict(item_counter)

    def calculate_metric(self, dataobject):
        item_matrix, item_count = self.used_info(dataobject)
        result = self.metric_info(self.get_pop(item_matrix, item_count))
        metric_dict = self.topk_result('averagepopularity', result)
        return metric_dict

    def get_pop(self, item_matrix, item_count):
        """Convert the matrix of item id to the matrix of item popularity using a dict:{id,count}.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            item_count(dict): the number of interaction of items in training data.

        Returns:
            numpy.ndarray: the popularity of items in the recommended list.
        """
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = item_count.get(row[j], 0)
        return value

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict


class ShannonEntropy(AbstractMetric):
    r"""ShannonEntropy_ presents the diversity of the recommendation items.
    It is the entropy over items' distribution.

    .. _ShannonEntropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.6700>`__
    and `paper <https://link.springer.com/article/10.1007/s13042-017-0762-9>`__

    .. math::
        \mathrm {ShannonEntropy@K}=-\sum_{i=1}^{|I|} p(i) \log p(i)

    :math:`p(i)` is the probability of recommending item i
    which is the number of item i in recommended list over all items.
    """
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items.
        """
        item_matrix = dataobject.get('rec.items')
        return item_matrix.numpy()

    def calculate_metric(self, dataobject):
        item_matrix = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = '{}@{}'.format('shannonentropy', k)
            metric_dict[key] = round(self.get_entropy(item_matrix[:, :k]), self.decimal_place)
        return metric_dict

    def get_entropy(self, item_matrix):
        """Get shannon entropy through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.

        Returns:
            float: the shannon entropy.
        """

        item_count = dict(Counter(item_matrix.flatten()))
        total_num = item_matrix.shape[0] * item_matrix.shape[1]
        result = 0.0
        for cnt in item_count.values():
            p = cnt / total_num
            result += -p * np.log(p)
        return result / len(item_count)


class GiniIndex(AbstractMetric):
    r"""GiniIndex presents the diversity of the recommendation items.
    It is used to measure the inequality of a distribution.

    .. _GiniIndex: https://en.wikipedia.org/wiki/Gini_coefficient

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3308560.3317303>`__.

    .. math::
        \mathrm {GiniIndex@K}=\left(\frac{\sum_{i=1}^{|I|}(2 i-|I|-1) P{(i)}}{|I| \sum_{i=1}^{|I|} P{(i)}}\right)

    :math:`P{(i)}` represents the number of times all items appearing in the recommended list,
    which is indexed in non-decreasing order (P_{(i)} \leq P_{(i+1)}).
    """
    metric_type = EvaluatorType.RANKING
    smaller = True
    metric_need = ['rec.items', 'data.num_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set"""
        item_matrix = dataobject.get('rec.items')
        num_items = dataobject.get('data.num_items')
        return item_matrix.numpy(), num_items

    def calculate_metric(self, dataobject):
        item_matrix, num_items = self.used_info(dataobject)
        metric_dict = {}
        for k in self.topk:
            key = '{}@{}'.format('giniindex', k)
            metric_dict[key] = round(self.get_gini(item_matrix[:, :k], num_items), self.decimal_place)
        return metric_dict

    def get_gini(self, item_matrix, num_items):
        """Get gini index through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            num_items(int): the total number of items.

        Returns:
            float: the gini index.
        """
        item_count = dict(Counter(item_matrix.flatten()))
        sorted_count = np.array(sorted(item_count.values()))
        num_recommended_items = sorted_count.shape[0]
        total_num = item_matrix.shape[0] * item_matrix.shape[1]
        idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
        gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / total_num
        gini_index /= num_items
        return gini_index


class TailPercentage(AbstractMetric):
    r"""TailPercentage_ computes the percentage of long-tail items in recommendation items.

    .. _TailPercentage: https://en.wikipedia.org/wiki/Long_tail#Criticisms

    For further details, please refer to the `paper <https://arxiv.org/pdf/2007.12329.pdf>`__.

    .. math::
        \mathrm {TailPercentage@K}=\frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in R_{u}} {\delta(i \in T)}}{|R_{u}|}

    :math:`\delta(·)` is an indicator function.
    :math:`T` is the set of long-tail items,
    which is a portion of items that appear in training data seldomly.

    Note:
        If you want to use this metric, please set the parameter 'tail_ratio' in the config
        which can be an integer or a float in (0,1]. Otherwise it will default to 0.1.
    """
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.count_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']
        self.tail = config['tail_ratio']
        if self.tail is None or self.tail <= 0:
            self.tail = 0.1

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set."""
        item_matrix = dataobject.get('rec.items')
        count_items = dataobject.get('data.count_items')
        return item_matrix.numpy(), dict(count_items)

    def get_tail(self, item_matrix, count_items):
        """Get long-tail percentage through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            count_items(dict): the number of interaction of items in training data.

        Returns:
            float: long-tail percentage.
        """
        if self.tail > 1:
            tail_items = [item for item, cnt in count_items.items() if cnt <= self.tail]
        else:
            count_items = sorted(count_items.items(), key=lambda kv: (kv[1], kv[0]))
            cut = max(int(len(count_items) * self.tail), 1)
            count_items = count_items[:cut]
            tail_items = [item for item, cnt in count_items]
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = 1 if row[j] in tail_items else 0
        return value

    def calculate_metric(self, dataobject):
        item_matrix, count_items = self.used_info(dataobject)
        result = self.metric_info(self.get_tail(item_matrix, count_items))
        metric_dict = self.topk_result('tailpercentage', result)
        return metric_dict

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict


class PopularityPercentage(AbstractMetric):
    """
    PopularityPercentage refers to the proportion of popular items in the recommendation list against the total number of items
    in the list, which can be seen as a popularity level measure of fairness.

    For further details, please refer to the paper https://doi.org/10.1145/3437963.3441824
    """
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.count_items']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']
        self.popularity = config['popularity_ratio']
        if self.popularity is None or self.popularity <= 0:
            self.popularity = 0.1

    def used_info(self, dataobject):
        """Get the matrix of recommendation items and number of items in total item set."""
        item_matrix = dataobject.get('rec.items')
        count_items = dataobject.get('data.count_items')
        return item_matrix.numpy(), dict(count_items)

    def get_popularity(self, item_matrix, count_items):
        """Get popularity percentage through the top-k recommendation list.

        Args:
            item_matrix(numpy.ndarray): matrix of items recommended to users.
            count_items(dict): the number of interaction of items in training data.

        Returns:
            float: popularity percentage.
        """
        if self.popularity > 1:
            tail_items = [item for item, cnt in count_items.items() if cnt >= self.popularity]
        else:
            count_items = sorted(count_items.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            cut = max(int(len(count_items) * self.popularity), 1)
            count_items = count_items[:cut]
            tail_items = [item for item, cnt in count_items]
        value = np.zeros_like(item_matrix)
        for i in range(item_matrix.shape[0]):
            row = item_matrix[i, :]
            for j in range(row.shape[0]):
                value[i][j] = 1 if row[j] in tail_items else 0
        return value

    def calculate_metric(self, dataobject):
        item_matrix, count_items = self.used_info(dataobject)
        result = self.metric_info(self.get_popularity(item_matrix, count_items))
        metric_dict = self.topk_result('popularitypercentage', result)
        return metric_dict

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict


class NonParityUnfairness(AbstractMetric):
    r"""NonParityUnFairness measures unfairness of non-parity

        For further details, please refer to the `paper <https://proceedings.neurips.cc/paper/2017/file/e6384711491713d29bc63fc5eeb5ba4f-Paper.pdf>`__.

        .. math::
            \mathrm {\left|\mathrm{E}_{g}[y]-\mathrm{E}_{\neg g}[y]\right|}

        :math:`g` is protected group.
        :math:`\neg g` is unprotected group.

        """
    smaller = True
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.positive_score', 'data.sst']

    def __init__(self, config):
        super().__init__(config)
        self.sst_attr_list = config['sst_attr_list']

    def used_info(self, dataobject):
        score = dataobject.get('rec.positive_score').numpy()
        sst_dict = {}
        for sst in self.sst_attr_list:
            sst_dict[sst] = dataobject.get('data.' + sst).numpy()

        return score, sst_dict

    def calculate_metric(self, dataobject):
        score, sst_dict = self.used_info(dataobject)
        metric_dict = {}
        for sst, value in sst_dict.items():
            key = 'NonParity Unfairness of sensitive attribute {}'.format(sst)
            metric_dict[key] = round(self.get_nonparity(score, sst, value), self.decimal_place)

        return metric_dict

    def get_nonparity(self, score, sst, sst_value):
        r"""

        Args:
            score(numpy.array): score prediction for user-item pairs
            sst(str): sensitive attribute
            sst_value(numpy.array): sensitive attribute's value of corresponding users
        Return:
            difference for sensitive attribute with binary value or std for multiple-value attribute
        """
        unique_value = np.unique(sst_value)
        if len(unique_value) < 2:
            raise ValueError(f'there is only one value for {sst} sensitive attribute')

        sst_avg_score = []
        for s in unique_value:
            sst_avg_score.append(np.mean(score[sst_value == s]))

        if len(unique_value) == 2:
            return np.abs(sst_avg_score[0] - sst_avg_score[1])
        else:
            return np.std(sst_avg_score)


class ValueUnfairness(AbstractMetric):
    r"""ValueUnfairness measures value unfairness of non-parity

        For further details, please refer to the `paper <https://proceedings.neurips.cc/paper/2017/file/e6384711491713d29bc63fc5eeb5ba4f-Paper.pdf>`__.

        .. math::
            \frac{1}{n} \sum_{j=1}^{n}\left|\left(\mathrm{E}_{g}[y]_{j}-\mathrm{E}_{g}[r]_{j}\right)-\left(\mathrm{E}_{\neg g}[y]_{j}-\mathrm{E}_{\neg g}[r]_{j}\right)\right|

            \mathrm{E}_{g}[y]_{j}:=\frac{1}{\left|\left\{i:((i, j) \in X) \wedge g_{i}\right\}\right|} \sum_{i:((i, j) \in X) \wedge g_{i}} y_{i j}

        :math:`g` is protected group.
        :math:`\neg g` is unprotected group.

        """
    smaller = True
    metric_type = EvaluatorType.RANKING

    metric_need = ['data.positive_i', 'rec.positive_score', 'data.negative_i', 'rec.negative_score', 'data.sst']

    def __init__(self, config):
        super().__init__(config)
        self.sst_key = config['sst_attr_list'][0]
        self.mode = config['eval_args']['mode']

    def used_info(self, dataobject):
        pos_score = dataobject.get('rec.positive_score').numpy()
        pos_iids = dataobject.get('data.positive_i').numpy()
        if self.mode != 'full':
            neg_score = dataobject.get('rec.negative_score').numpy()
            neg_iids = dataobject.get('data.negative_i').numpy()
        sst_value = dataobject.get('data.' + self.sst_key).numpy()
        if self.mode != 'full':
            return pos_score, pos_iids, neg_score, neg_iids, sst_value
        else:
            return pos_score, pos_iids, sst_value

    def calculate_metric(self, dataobject):
        if self.mode != 'full':
            pos_score, pos_iids, neg_score, neg_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Value Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_value_unfairness(pos_score, pos_iids, neg_score, neg_iids, sst_value),
                                     self.decimal_place)
        else:
            pos_score, pos_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Value Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_value_unfairness(pos_score, pos_iids, None, None, sst_value),
                                     self.decimal_place)
        return metric_dict

    def get_value_unfairness(self, pos_score, pos_iids, neg_score, neg_iids, sst_value):
        r"""

        Args:
            score(numpy.array): score prediction for user-item pairs
            iids(numpy.array): item_id array of interaction ITEM_FIELD
            sst_value(numpy.array): sensitive attribute's value of corresponding users
        Return:
            Value Unfairness
        """
        sst_unique_values, sst_indices = np.unique(sst_value, return_inverse=True)
        if self.mode != 'full':
            iid_unique_values, iid_indices = np.unique(np.concatenate((pos_iids, neg_iids)), return_inverse=True)
        else:
            iid_unique_values, iid_indices = np.unique(pos_iids, return_inverse=True)

        if len(sst_unique_values) != 2:
            raise ValueError(f'sensitive attribute must be binary')

        pos_len = len(pos_iids)
        iids_len = len(iid_unique_values)
        avg_pred_list = np.zeros((iids_len, 2))
        sst_num = np.zeros((iids_len, 2))
        avg_true_list = np.zeros((iids_len, 2))

        for iid_indice, sst_indice, score in zip(iid_indices[:pos_len], sst_indices, pos_score):
            avg_pred_list[iid_indice][sst_indice] += score
            sst_num[iid_indice][sst_indice] += 1
            avg_true_list[iid_indice][sst_indice] += 1

        if self.mode != 'full':
            for iid_indice, sst_indice, score in zip(iid_indices[pos_len:], sst_indices, neg_score):
                avg_pred_list[iid_indice][sst_indice] += score
                sst_num[iid_indice][sst_indice] += 1

        sst_num += 1e-5

        avg_pred_list /= sst_num
        avg_true_list /= sst_num

        diff = avg_pred_list - avg_true_list
        diff = np.mean(np.abs(diff[:, 0] - diff[:, 1]))

        return diff


class AbsoluteUnfairness(AbstractMetric):
    r"""AbsoluteUnfairness measures absolute unfairness

        For further details, please refer to the `paper <https://proceedings.neurips.cc/paper/2017/file/e6384711491713d29bc63fc5eeb5ba4f-Paper.pdf>`__.

        .. math::
            \frac{1}{n} \sum_{j=1}^{n}\left\|\left|\mathrm{E}_{g}[y]_{j}-\mathrm{E}_{g}[r]_{j}\right|-\mid \mathrm{E}_{\neg g}[y]_{j}-\mathrm{E}_{\neg g}[r]_{j}\right\|

            \mathrm{E}_{g}[y]_{j}:=\frac{1}{\left|\left\{i:((i, j) \in X) \wedge g_{i}\right\}\right|} \sum_{i:((i, j) \in X) \wedge g_{i}} y_{i j}

        :math:`g` is protected group.
        :math:`\neg g` is unprotected group.

        """
    smaller = True
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.positive_i', 'rec.positive_score', 'data.negative_i', 'rec.negative_score', 'data.sst']

    def __init__(self, config):
        super().__init__(config)
        self.sst_key = config['sst_attr_list'][0]
        self.mode = config['eval_args']['mode']

    def used_info(self, dataobject):
        pos_score = dataobject.get('rec.positive_score').numpy()
        pos_iids = dataobject.get('data.positive_i').numpy()
        if self.mode != 'full':
            neg_score = dataobject.get('rec.negative_score').numpy()
            neg_iids = dataobject.get('data.negative_i').numpy()
        sst_value = dataobject.get('data.' + self.sst_key).numpy()
        if self.mode != 'full':
            return pos_score, pos_iids, neg_score, neg_iids, sst_value
        else:
            return pos_score, pos_iids, sst_value

    def calculate_metric(self, dataobject):
        if self.mode != 'full':
            pos_score, pos_iids, neg_score, neg_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Absolute Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_absolute_unfairness(pos_score, pos_iids, neg_score, neg_iids, sst_value),
                                     self.decimal_place)
        else:
            pos_score, pos_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Absolute Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_absolute_unfairness(pos_score, pos_iids, None, None, sst_value),
                                     self.decimal_place)
        return metric_dict

    def get_absolute_unfairness(self, pos_score, pos_iids, neg_score, neg_iids, sst_value):
        r"""

        Args:
            score(numpy.array): score prediction for user-item pairs
            iids(numpy.array): item_id array of interaction ITEM_FIELD
            sst_value(numpy.array): sensitive attribute's value of corresponding users
        Return:
            Absolute Unfairness
        """
        sst_unique_values, sst_indices = np.unique(sst_value, return_inverse=True)
        if self.mode != 'full':
            iid_unique_values, iid_indices = np.unique(np.concatenate((pos_iids, neg_iids)), return_inverse=True)
        else:
            iid_unique_values, iid_indices = np.unique(pos_iids, return_inverse=True)

        if len(sst_unique_values) != 2:
            raise ValueError(f'sensitive attribute must be binary')

        pos_len = len(pos_iids)
        iids_len = len(iid_unique_values)
        avg_pred_list = np.zeros((iids_len, 2))
        sst_num = np.zeros((iids_len, 2))
        avg_true_list = np.zeros((iids_len, 2))

        for iid_indice, sst_indice, score in zip(iid_indices[:pos_len], sst_indices, pos_score):
            avg_pred_list[iid_indice][sst_indice] += score
            sst_num[iid_indice][sst_indice] += 1
            avg_true_list[iid_indice][sst_indice] += 1

        if self.mode != 'full':
            for iid_indice, sst_indice, score in zip(iid_indices[pos_len:], sst_indices, neg_score):
                avg_pred_list[iid_indice][sst_indice] += score
                sst_num[iid_indice][sst_indice] += 1

        sst_num += 1e-5

        avg_pred_list /= sst_num
        avg_true_list /= sst_num

        diff = np.abs(avg_pred_list - avg_true_list)
        diff = np.mean(np.abs(diff[:, 0] - diff[:, 1]))

        return diff


class UnderUnfairness(AbstractMetric):
    r"""UnderUnfairness measures underestimation unfairness

        For further details, please refer to the `paper <https://proceedings.neurips.cc/paper/2017/file/e6384711491713d29bc63fc5eeb5ba4f-Paper.pdf>`__.

        .. math::
            \frac{1}{n} \sum_{j=1}^{n}\left|\max \left\{0, \mathrm{E}_{g}[r]_{j}-\mathrm{E}_{g}[y]_{j}\right\}-\max \left\{0, \mathrm{E}_{\neg g}[r]_{j}-\mathrm{E}_{\neg g}[y]_{j}\right\}\right|

            \mathrm{E}_{g}[y]_{j}:=\frac{1}{\left|\left\{i:((i, j) \in X) \wedge g_{i}\right\}\right|} \sum_{i:((i, j) \in X) \wedge g_{i}} y_{i j}

        :math:`g` is protected group.
        :math:`\neg g` is unprotected group.

        """
    smaller = True
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.positive_i', 'rec.positive_score', 'data.negative_i', 'rec.negative_score', 'data.sst']

    def __init__(self, config):
        super().__init__(config)
        self.sst_key = config['sst_attr_list'][0]
        self.mode = config['eval_args']['mode']

    def used_info(self, dataobject):
        pos_score = dataobject.get('rec.positive_score').numpy()
        pos_iids = dataobject.get('data.positive_i').numpy()
        if self.mode != 'full':
            neg_score = dataobject.get('rec.negative_score').numpy()
            neg_iids = dataobject.get('data.negative_i').numpy()
        sst_value = dataobject.get('data.' + self.sst_key).numpy()
        if self.mode != 'full':
            return pos_score, pos_iids, neg_score, neg_iids, sst_value
        else:
            return pos_score, pos_iids, sst_value

    def calculate_metric(self, dataobject):
        if self.mode != 'full':
            pos_score, pos_iids, neg_score, neg_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Underestimation Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_under_unfairness(pos_score, pos_iids, neg_score, neg_iids, sst_value),
                                     self.decimal_place)
        else:
            pos_score, pos_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Underestimation Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_under_unfairness(pos_score, pos_iids, None, None, sst_value),
                                     self.decimal_place)
        return metric_dict

    def get_under_unfairness(self, pos_score, pos_iids, neg_score, neg_iids, sst_value):
        r"""

        Args:
            score(numpy.array): score prediction for user-item pairs
            iids(numpy.array): item_id array of interaction ITEM_FIELD
            sst_value(numpy.array): sensitive attribute's value of corresponding users
        Return:
            Underestimation Unfairness
        """
        sst_unique_values, sst_indices = np.unique(sst_value, return_inverse=True)
        if self.mode != 'full':
            iid_unique_values, iid_indices = np.unique(np.concatenate((pos_iids, neg_iids)), return_inverse=True)
        else:
            iid_unique_values, iid_indices = np.unique(pos_iids, return_inverse=True)

        if len(sst_unique_values) != 2:
            raise ValueError(f'sensitive attribute must be binary')

        pos_len = len(pos_iids)
        iids_len = len(iid_unique_values)
        avg_pred_list = np.zeros((iids_len, 2))
        sst_num = np.zeros((iids_len, 2))
        avg_true_list = np.zeros((iids_len, 2))

        for iid_indice, sst_indice, score in zip(iid_indices[:pos_len], sst_indices, pos_score):
            avg_pred_list[iid_indice][sst_indice] += score
            sst_num[iid_indice][sst_indice] += 1
            avg_true_list[iid_indice][sst_indice] += 1

        if self.mode != 'full':
            for iid_indice, sst_indice, score in zip(iid_indices[pos_len:], sst_indices, neg_score):
                avg_pred_list[iid_indice][sst_indice] += score
                sst_num[iid_indice][sst_indice] += 1

        sst_num += 1e-5

        avg_pred_list /= sst_num
        avg_true_list /= sst_num

        diff = np.where((avg_true_list - avg_pred_list) > 0, avg_true_list - avg_pred_list, 0)
        diff = np.mean(np.abs(diff[:, 0] - diff[:, 1]))

        return diff


class OverUnfairness(AbstractMetric):
    r"""OverUnfairness measures overestimation unfairness

        For further details, please refer to the `paper <https://proceedings.neurips.cc/paper/2017/file/e6384711491713d29bc63fc5eeb5ba4f-Paper.pdf>`__.

        .. math::
            \frac{1}{n} \sum_{j=1}^{n}\left|\max \left\{0, \mathrm{E}_{g}[r]_{j}-\mathrm{E}_{g}[y]_{j}\right\}-\max \left\{0, \mathrm{E}_{\neg g}[r]_{j}-\mathrm{E}_{\neg g}[y]_{j}\right\}\right|

            \mathrm{E}_{g}[y]_{j}:=\frac{1}{\left|\left\{i:((i, j) \in X) \wedge g_{i}\right\}\right|} \sum_{i:((i, j) \in X) \wedge g_{i}} y_{i j}

        :math:`g` is protected group.
        :math:`\neg g` is unprotected group.

        """
    smaller = True
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.positive_i', 'rec.positive_score', 'data.negative_i', 'rec.negative_score', 'data.sst']

    def __init__(self, config):
        super().__init__(config)
        self.sst_key = config['sst_attr_list'][0]
        self.mode = config['eval_args']['mode']

    def used_info(self, dataobject):
        pos_score = dataobject.get('rec.positive_score').numpy()
        pos_iids = dataobject.get('data.positive_i').numpy()
        if self.mode != 'full':
            neg_score = dataobject.get('rec.negative_score').numpy()
            neg_iids = dataobject.get('data.negative_i').numpy()
        sst_value = dataobject.get('data.' + self.sst_key).numpy()

        if self.mode != 'full':
            return pos_score, pos_iids, neg_score, neg_iids, sst_value
        else:
            return pos_score, pos_iids, sst_value

    def calculate_metric(self, dataobject):
        if self.mode != 'full':
            pos_score, pos_iids, neg_score, neg_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Overestimation Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_over_unfairness(pos_score, pos_iids, neg_score, neg_iids, sst_value),
                                     self.decimal_place)
        else:
            pos_score, pos_iids, sst_value = self.used_info(dataobject)
            metric_dict = {}
            key = 'Overestimation Unfairness of sensitive attribute {}'.format(self.sst_key)
            metric_dict[key] = round(self.get_over_unfairness(pos_score, pos_iids, None, None, sst_value),
                                     self.decimal_place)
        return metric_dict

    def get_over_unfairness(self, pos_score, pos_iids, neg_score, neg_iids, sst_value):
        r"""

        Args:
            score(numpy.array): score prediction for user-item pairs
            iids(numpy.array): item_id array of interaction ITEM_FIELD
            sst_value(numpy.array): sensitive attribute's value of corresponding users
        Return:
            Overestimation Unfairness
        """
        sst_unique_values, sst_indices = np.unique(sst_value, return_inverse=True)
        if self.mode != 'full':
            iid_unique_values, iid_indices = np.unique(np.concatenate((pos_iids, neg_iids)), return_inverse=True)
        else:
            iid_unique_values, iid_indices = np.unique(pos_iids, return_inverse=True)
        if len(sst_unique_values) != 2:
            raise ValueError(f'sensitive attribute must be binary')

        pos_len = len(pos_iids)
        iids_len = len(iid_unique_values)
        avg_pred_list = np.zeros((iids_len, 2))
        sst_num = np.zeros((iids_len, 2))
        avg_true_list = np.zeros((iids_len, 2))

        for iid_indice, sst_indice, score in zip(iid_indices[:pos_len], sst_indices, pos_score):
            avg_pred_list[iid_indice][sst_indice] += score
            sst_num[iid_indice][sst_indice] += 1
            avg_true_list[iid_indice][sst_indice] += 1

        if self.mode != 'full':
            for iid_indice, sst_indice, score in zip(iid_indices[pos_len:], sst_indices, neg_score):
                avg_pred_list[iid_indice][sst_indice] += score
                sst_num[iid_indice][sst_indice] += 1

        sst_num += 1e-5

        avg_pred_list /= sst_num
        avg_true_list /= sst_num

        diff = np.where((avg_pred_list - avg_true_list) > 0, avg_pred_list - avg_true_list, 0)
        diff = np.mean(np.abs(diff[:, 0] - diff[:, 1]))

        return diff


class DifferentialFairness(AbstractMetric):
    """
    The DifferentialFairness metric aims to ensure equitable treatment for all protected groups.

    For further details, please refer to the https://dl.acm.org/doi/10.1145/3442381.3449904

    For gender bias in our recommender (assuming a gender binary), we can estimate epsilon-DF per sensitive item i by verifying that:

    .. math::
             \begin{gathered}
        e^{-\epsilon} \leq \frac{\sum_{u: A=m} \hat{y}_{u i}+\alpha}{N_{m}+2 \alpha} \frac{N_{f}+2 \alpha}{\sum_{u: A=f} \hat{y}_{u i}+\alpha} \leq e^{\epsilon} \\
        e^{-\epsilon} \leq \frac{\sum_{u: A=m}\left(1-\hat{y}_{u i}\right)+\alpha}{N_{m}+2 \alpha} \frac{N_{f}+2 \alpha}{\sum_{u: A=f}\left(1-\hat{y}_{u i}\right)+\alpha} \leq e^{\epsilon},
        \end{gathered}
    :math:`\alpha` is each entry of the parameter of a symmetric Dirichlet prior with concentration parameter 2\alpha.
    :math:`i` is an item.
    :math:`N_A` is the number of users of gender A (m or f ).

    """
    smaller = True
    metric_type = EvaluatorType.RANKING
    metric_need = ['data.positive_i', 'rec.positive_score', 'data.sst']

    def __init__(self, config):
        super().__init__(config)
        self.sst_key_list = config['sst_attr_list']

    def used_info(self, dataobject):
        score = dataobject.get('rec.positive_score').numpy()
        iids = dataobject.get('data.positive_i').numpy()
        sst_value_dict = {}
        for sst_key in self.sst_key_list:
            sst_value_dict[sst_key] = dataobject.get('data.' + sst_key).numpy()

        return score, iids, sst_value_dict

    def calculate_metric(self, dataobject):
        score, iids, sst_value_dict = self.used_info(dataobject)
        metric_dict = {}
        for sst_key, sst_value in sst_value_dict.items():
            key = 'Differential Fairness of sensitive attribute {}'.format(sst_key)
            metric_dict[key] = round(self.get_differential_fairness(score, iids, sst_value), self.decimal_place)

        return metric_dict

    def get_differential_fairness(self, score, iids, sst_value):
        r"""

        Args:
            score(numpy.array): score prediction for user-item pairs
            iids(numpy.array): item_id array of interaction ITEM_FIELD
            sst_value(numpy.array): sensitive attribute's value of corresponding users/items
        Return:
            Differential Fairness
        """
        sst_unique_values, sst_indices = np.unique(sst_value, return_inverse=True)
        iid_unique_values, iid_indices = np.unique(iids, return_inverse=True)
        score_matric = np.zeros((len(iid_unique_values), len(sst_unique_values)), dtype=np.float32)
        epsilon_values = np.zeros(len(iid_unique_values), dtype=np.float32)

        concentration_parameter = 1.0
        dirichlet_alpha = concentration_parameter / len(iid_unique_values)

        for i in range(len(iid_unique_values)):
            for j in range(len(sst_unique_values)):
                indices = (iid_indices == i) * (sst_indices == j)
                score_matric[i, j] = (score[indices].sum() + dirichlet_alpha) / (
                            indices.sum() + concentration_parameter)

        for i in range(len(sst_unique_values)):
            for j in range(i + 1, len(sst_unique_values)):
                epsilon = np.abs(np.log(score_matric[:, i]) - np.log(score_matric[:, j]))
                epsilon_values = np.where(epsilon > epsilon_values, epsilon, epsilon_values)

        return epsilon_values.mean()