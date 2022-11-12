import torchmetrics
from torchmetrics.functional import calibration_error
import torch
import torch.nn.functional as F


class Accuracy:
    def __init__(self):
        self.__name__ = 'accuracy'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        _, preds = probs.max(-1)
        return (preds == labels).float().mean()


class TopkAccuracy:
    def __init__(self, top_k=5):
        self.top_k = top_k
        if top_k == 1:
            self.__name__ = 'accuracy'
        else:
            self.__name__ = f"top_{top_k}_accuracy"

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        return torchmetrics.functional.classification.accuracy(probs, labels, top_k=self.top_k)


class Brier:
    def __init__(self, num_classes=-1):
        self.num_classes = num_classes
        self.__name__ = 'brier'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=self.num_classes)
        return ((probs - labels_onehot) ** 2).sum(-1).mean()


class ECE:
    def __init__(self, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.__name__ = 'ece'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        return calibration_error(probs, labels, n_bins=self.n_bins, norm='l1')


class UCE:
    def __init__(self, num_classes=-1, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.num_classes = torch.tensor(num_classes, dtype=float)
        self.__name__ = 'uce'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        preds = probs.argmax(-1)
        if self.num_classes == -1:
            num_classes = torch.tensor(F.one_hot(labels.to(torch.int64)).shape[-1], dtype=float)
        else:
            num_classes = self.num_classes
        uncertainties = 1/torch.log(num_classes) * entropy(probs)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(uncertainties, bins, right=True)

        bin_errors = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_uncertainties = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_counts = torch.zeros(self.n_bins, dtype=int, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_errors[b] = torch.mean((labels[selected] != preds[selected]).float())
                bin_uncertainties[b] = torch.mean(uncertainties[selected])
                bin_counts[b] = len(selected)

        gaps = torch.abs(bin_errors - bin_uncertainties)
        uce = torch.sum(gaps * bin_counts) / torch.sum(bin_counts)
        return uce


class MCE:
    def __init__(self, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.__name__ = 'mce'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        return calibration_error(probs, labels, n_bins=self.n_bins, norm='max')


class MUCE:
    def __init__(self, num_classes=-1, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.num_classes = torch.tensor(num_classes, dtype=float)
        self.__name__ = 'muce'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        preds = probs.argmax(-1)
        if self.num_classes == -1:
            num_classes = torch.tensor(F.one_hot(labels.to(torch.int64)).shape[-1], dtype=float)
        else:
            num_classes = self.num_classes
        uncertainties = 1/torch.log(num_classes) * entropy(probs)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(uncertainties, bins, right=True)

        bin_errors = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_uncertainties = torch.zeros(self.n_bins, dtype=float, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_errors[b] = torch.mean((labels[selected] != preds[selected]).float())
                bin_uncertainties[b] = torch.mean(uncertainties[selected])

        gaps = torch.abs(bin_errors - bin_uncertainties)
        muce = torch.max(gaps)
        return muce


class ACE:
    def __init__(self, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.__name__ = 'ace'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        probs = probs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        return torch.tensor(adaptive_calibration_error(labels, probs, self.n_bins))


class SCE:
    def __init__(self, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.__name__ = 'sce'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        if len(probs.shape) == 3:
            probs = probs.mean(0)
        if len(probs.shape) == 4:
            probs = probs.mean(1).mean(0)
        probs = probs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        return torch.tensor(static_calibration_error(labels, probs, self.n_bins))


class ModelUncertainty:
    def __init__(self):
        self.__name__ = 'model_uncertainty'

    def __call__(self, logits, labels=None):
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        pred_uncert = predictive_uncertainty(logits)
        data_uncert = data_uncertainty(logits)
        model_uncert = pred_uncert - data_uncert
        return model_uncert.mean()


class PredictiveUncertainty:
    def __init__(self):
        self.__name__ = 'predictive_uncertainty'

    def __call__(self, logits, labels=None):
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        return predictive_uncertainty(logits).mean()


class DataUncertainty:
    def __init__(self):
        self.__name__ = 'data_uncertainty'

    def __call__(self, logits, labels=None):
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        # probs.shape = (mc_samples, batch_size, num_classes)
        # return Tensor of shape (batch_size,)
        return data_uncertainty(logits).mean()


def entropy(probs, labels=None):
    return -torch.sum(probs * torch.log(probs + torch.finfo().tiny), axis=-1)


def model_uncertainty(logits, labels=None):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    pred_uncert = predictive_uncertainty(logits)
    data_uncert = data_uncertainty(logits)
    return pred_uncert - data_uncert


def predictive_uncertainty(logits, labels=None):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(-1)
    if len(probs.shape) == 4:
        probs = probs.mean(1)
    return entropy(probs.mean(0))


def data_uncertainty(logits, labels=None):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(-1)
    if len(probs.shape) == 4:
        probs = probs.mean(1)
    return entropy(probs).mean(0)


# Reference GeneralCalibrationError: https://github.com/JeremyNixon/uncertainty-metrics-1
# coding=utf-8
# Copyright 2021 The Uncertainty Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""General metric defining the parameterized space of calibration metrics.
"""

import itertools
import numpy as np


def one_hot_encode(labels, num_classes=None):
    """One hot encoder for turning a vector of labels into a OHE matrix."""
    if num_classes is None:
        num_classes = len(np.unique(labels))
    return np.eye(num_classes)[labels]


def mean(inputs):
    """Be able to take the mean of an empty array without hitting NANs."""
    # pylint disable necessary for numpy and pandas
    if len(inputs) == 0:  # pylint: disable=g-explicit-length-test
        return 0
    else:
        return np.mean(inputs)


def get_adaptive_bins(predictions, num_bins):
    """Returns upper edges for binning an equal number of datapoints per bin."""
    if np.size(predictions) == 0:
        return np.linspace(0, 1, num_bins+1)[:-1]

    edge_indices = np.linspace(0, len(predictions), num_bins, endpoint=False)

    # Round into integers for indexing. If num_bins does not evenly divide
    # len(predictions), this means that bin sizes will alternate between SIZE and
    # SIZE+1.
    edge_indices = np.round(edge_indices).astype(int)

    # If there are many more bins than data points, some indices will be
    # out-of-bounds by one. Set them to be within bounds:
    edge_indices = np.minimum(edge_indices, len(predictions) - 1)

    # Obtain the edge values:
    edges = np.sort(predictions)[edge_indices]

    # Following the convention of numpy.digitize, we do not include the leftmost
    # edge (i.e. return the upper bin edges):
    return edges[1:]


def binary_converter(probs):
    """Converts a binary probability vector into a matrix."""
    return np.array([[1-p, p] for p in probs])


class GeneralCalibrationError():
    """Implements the space of calibration errors, General Calibration Error.

    This implementation of General Calibration Error can be class-conditional,
    adaptively binned, thresholded, focus on the maximum or top labels, and use
    the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
    definitions of most of these terms, see [1].

    To implement Expected Calibration Error [2]:
    ECE = GeneralCalibrationError(binning_scheme='even', class_conditional=False,
      max_prob=True, error='l1')

    To implement Static Calibration Error [1]:
    SCE = GeneralCalibrationError(binning_scheme='even', class_conditional=False,
      max_prob=False, error='l1')

    To implement Root Mean Squared Calibration Error [3]:
    RMSCE = GeneralCalibrationError(binning_scheme='adaptive',
    class_conditional=False, max_prob=True, error='l2', datapoints_per_bin=100)

    To implement Adaptive Calibration Error [1]:
    ACE = GeneralCalibrationError(binning_scheme='adaptive',
    class_conditional=True, max_prob=False, error='l1')

    To implement Thresholded Adaptive Calibration Error [1]:
    TACE = GeneralCalibrationError(binning_scheme='adaptive',
    class_conditional=True, max_prob=False, error='l1', threshold=0.01)

    ### References

    [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
    and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
    the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
    pp. 38-41. 2019.
    https://arxiv.org/abs/1904.01685

    [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
    "Obtaining well calibrated probabilities using bayesian binning."
    Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/

    [3] Khanh Nguyen and Brendan O’Connor.
    "Posterior calibration and exploratory analysis for natural language
    processing models."  Empirical Methods in Natural Language Processing. 2015.
    https://arxiv.org/pdf/1508.05154.pdf

    Attributes:
      binning_scheme: String, either 'even' (for even spacing) or 'adaptive'
        (for an equal number of datapoints in each bin).

      max_prob: Boolean, 'True' to measure calibration only on the maximum
        prediction for each datapoint, 'False' to look at all predictions.

      class_conditional: Boolean, 'False' for the case where predictions from
        different classes are binned together, 'True' for binned separately.

      norm: String, apply 'l1' or 'l2' norm to the calibration error.

      num_bins: Integer, number of bins of confidence scores to use.

      threshold: Float, only look at probabilities above a certain value.

      datapoints_per_bin: Int, number of datapoints in each adaptive bin. This
        is a second option when binning adaptively - you can use either num_bins
        or this method to determine the bin size.

      distribution: String, data distribution this metric is measuring, whether
        train, test, out-of-distribution, or the user's choice.

      accuracies: Vector, accuracy within each bin.

      confidences: Vector, mean confidence within each bin.

      calibration_error: Float, computed calibration error.

      calibration_errors: Vector, difference between accuracies and confidences.
    """

    def __init__(self,
                 binning_scheme,
                 max_prob,
                 class_conditional,
                 norm,
                 num_bins=30,
                 threshold=0.0,
                 datapoints_per_bin=None,
                 distribution=None):
        self.binning_scheme = binning_scheme
        self.max_prob = max_prob
        self.class_conditional = class_conditional
        self.norm = norm
        self.num_bins = num_bins
        self.threshold = threshold
        self.datapoints_per_bin = datapoints_per_bin
        self.distribution = distribution
        self.accuracies = None
        self.confidences = None
        self.calibration_error = None
        self.calibration_errors = None

    def get_calibration_error(self, probs, labels, bin_upper_bounds, norm,
                              num_bins):
        """Given a binning scheme, returns sum weighted calibration error."""
        if np.size(probs) == 0:
            return 0.

        bin_indices = np.digitize(probs, bin_upper_bounds)
        sums = np.bincount(bin_indices, weights=probs, minlength=num_bins)
        sums = sums.astype(np.float64)  # In case all probs are 0/1.
        counts = np.bincount(bin_indices, minlength=num_bins)
        counts = counts + np.finfo(sums.dtype).eps  # Avoid division by zero.
        self.confidences = sums / counts
        self.accuracies = np.bincount(
            bin_indices, weights=labels, minlength=num_bins) / counts

        self.calibration_errors = self.accuracies-self.confidences

        if norm == 'l1':
            calibration_errors_normed = self.calibration_errors
        elif norm == 'l2':
            calibration_errors_normed = np.square(self.calibration_errors)
        else:
            raise ValueError(f'Unknown norm: {norm}')

        weighting = counts / float(len(probs.flatten()))
        weighted_calibration_error = calibration_errors_normed * weighting

        return np.sum(np.abs(weighted_calibration_error))

    def update_state(self, labels, probs):
        """Updates the value of the General Calibration Error."""

        # if self.calibration_error is not None and

        probs = np.array(probs)
        labels = np.array(labels)
        if probs.ndim == 2:

            num_classes = probs.shape[1]
            if num_classes == 1:
                probs = probs[:, 0]
                probs = binary_converter(probs)
                num_classes = 2
        elif probs.ndim == 1:
            # Cover binary case
            probs = binary_converter(probs)
            num_classes = 2
        else:
            raise ValueError('Probs must have 1 or 2 dimensions.')

        # Convert the labels vector into a one-hot-encoded matrix.
        labels_matrix = one_hot_encode(labels, probs.shape[1])

        if self.datapoints_per_bin is not None:
            self.num_bins = int(len(probs)/self.datapoints_per_bin)
            if self.binning_scheme != 'adaptive':
                raise ValueError(
                    "To set datapoints_per_bin, binning_scheme must be 'adaptive'.")

        if self.binning_scheme == 'even':
            bin_upper_bounds = np.histogram_bin_edges(
                [], bins=self.num_bins, range=(0.0, 1.0))[1:]

        # When class_conditional is False, different classes are conflated.
        if not self.class_conditional:
            if self.max_prob:
                labels_matrix = labels_matrix[
                    range(len(probs)), np.argmax(probs, axis=1)]
                probs = probs[range(len(probs)), np.argmax(probs, axis=1)]
            labels_matrix = labels_matrix[probs > self.threshold]
            probs = probs[probs > self.threshold]
            if self.binning_scheme == 'adaptive':
                bin_upper_bounds = get_adaptive_bins(probs, self.num_bins)
            calibration_error = self.get_calibration_error(
                probs.flatten(), labels_matrix.flatten(), bin_upper_bounds, self.norm,
                self.num_bins)

        # If class_conditional is true, predictions from different classes are
        # binned separately.
        else:
            # Initialize list for class calibration errors.
            class_calibration_error_list = []
            for j in range(num_classes):
                if not self.max_prob:
                    probs_slice = probs[:, j]
                    labels = labels_matrix[:, j]
                    labels = labels[probs_slice > self.threshold]
                    probs_slice = probs_slice[probs_slice > self.threshold]
                    if self.binning_scheme == 'adaptive':
                        bin_upper_bounds = get_adaptive_bins(probs_slice, self.num_bins)
                    calibration_error = self.get_calibration_error(
                        probs_slice, labels, bin_upper_bounds, self.norm, self.num_bins)
                    class_calibration_error_list.append(calibration_error/num_classes)
                else:
                    # In the case where we use all datapoints,
                    # max label has to be applied before class splitting.
                    labels = labels_matrix[np.argmax(probs, axis=1) == j][:, j]
                    probs_slice = probs[np.argmax(probs, axis=1) == j][:, j]
                    labels = labels[probs_slice > self.threshold]
                    probs_slice = probs_slice[probs_slice > self.threshold]
                    if self.binning_scheme == 'adaptive':
                        bin_upper_bounds = get_adaptive_bins(probs_slice, self.num_bins)
                    calibration_error = self.get_calibration_error(
                        probs_slice, labels, bin_upper_bounds, self.norm, self.num_bins)
                    class_calibration_error_list.append(calibration_error/num_classes)
            calibration_error = np.sum(class_calibration_error_list)

        if self.norm == 'l2':
            calibration_error = np.sqrt(calibration_error)

        self.calibration_error = calibration_error

    def result(self):
        return self.calibration_error

    def reset_state(self):
        self.calibration_error = None


def general_calibration_error(labels,
                              probs,
                              binning_scheme,
                              max_prob,
                              class_conditional,
                              norm,
                              num_bins=30,
                              threshold=0.0,
                              datapoints_per_bin=None):
    """Implements the space of calibration errors, General Calibration Error.

    This implementation of General Calibration Error can be class-conditional,
    adaptively binned, thresholded, focus on the maximum or top labels, and use
    the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
    definitions of most of these terms, see [1].

    To implement Expected Calibration Error [2]:
    gce(labels, probs, binning_scheme='even', class_conditional=False,
      max_prob=True, error='l1')

    To implement Static Calibration Error [1]:
    gce(labels, probs, binning_scheme='even', class_conditional=False,
      max_prob=False, error='l1')

    To implement Root Mean Squared Calibration Error [3]:
    gce(labels, probs, binning_scheme='adaptive', class_conditional=False,
      max_prob=True, error='l2', datapoints_per_bin=100)

    To implement Adaptive Calibration Error [1]:
    gce(labels, probs, binning_scheme='adaptive', class_conditional=True,
      max_prob=False, error='l1')

    To implement Thresholded Adaptive Calibration Error [1]:
    gce(labels, probs, binning_scheme='adaptive', class_conditional=True,
      max_prob=False, error='l1', threshold=0.01)

    ### References

    [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
    and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
    the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
    pp. 38-41. 2019.
    https://arxiv.org/abs/1904.01685

    [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
    "Obtaining well calibrated probabilities using bayesian binning."
    Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/

    [3] Khanh Nguyen and Brendan O’Connor.
    "Posterior calibration and exploratory analysis for natural language
    processing models."  Empirical Methods in Natural Language Processing. 2015.
    https://arxiv.org/pdf/1508.05154.pdf

    Args:
      labels: np.ndarray of shape [N, ] array of correct labels.
      probs: np.ndarray of shape [N, M] where N is the number of datapoints
        and M is the number of predicted classes.
      binning_scheme: String, either 'even' (for even spacing) or 'adaptive'
        (for an equal number of datapoints in each bin).
      max_prob: Boolean, 'True' to measure calibration only on the maximum
        prediction for each datapoint, 'False' to look at all predictions.
      class_conditional: Boolean, 'False' for the case where predictions from
        different classes are binned together, 'True' for binned separately.
      norm: String, apply 'l1' or 'l2' norm to the calibration error.
      num_bins: Integer, number of bins of confidence scores to use.
      threshold: Float, only look at probabilities above a certain value.
      datapoints_per_bin: Int, number of datapoints in each adaptive bin. This
        is a second option when binning adaptively - you can use either num_bins
        or this method to determine the bin size.

    Raises:
      ValueError.

    Returns:
      Float, general calibration error.

    """
    metric = GeneralCalibrationError(num_bins=num_bins,
                                     binning_scheme=binning_scheme,
                                     class_conditional=class_conditional,
                                     max_prob=max_prob,
                                     norm=norm,
                                     threshold=threshold,
                                     datapoints_per_bin=datapoints_per_bin)
    metric.update_state(labels, probs)
    return metric.result()


def expected_calibration_error(labels, probs, num_bins=30):
    """Implements Expected Calibration Error."""
    return general_calibration_error(labels,
                                     probs,
                                     binning_scheme='even',
                                     max_prob=True,
                                     class_conditional=False,
                                     norm='l1',
                                     num_bins=num_bins)


def root_mean_squared_calibration_error(labels, probs, num_bins=30, datapoints_per_bin=None):
    """Implements Root Mean Squared Calibration Error."""
    return general_calibration_error(labels,
                                     probs,
                                     binning_scheme='adaptive',
                                     max_prob=True,
                                     class_conditional=False,
                                     norm='l2',
                                     num_bins=num_bins,
                                     datapoints_per_bin=datapoints_per_bin)


def static_calibration_error(labels, probs, num_bins=30):
    """Implements Static Calibration Error."""
    return general_calibration_error(labels,
                                     probs,
                                     binning_scheme='even',
                                     max_prob=False,
                                     class_conditional=True,
                                     norm='l1',
                                     num_bins=num_bins)


def adaptive_calibration_error(labels, probs, num_bins=30):
    """Implements Adaptive Calibration Error."""
    return general_calibration_error(labels,
                                     probs,
                                     binning_scheme='adaptive',
                                     max_prob=False,
                                     class_conditional=True,
                                     norm='l1',
                                     num_bins=num_bins)


def thresholded_adaptive_calibration_error(labels, probs, num_bins=30, threshold=0.01):
    """Implements Thresholded Adaptive Calibration Error."""
    return general_calibration_error(labels,
                                     probs,
                                     binning_scheme='adaptive',
                                     max_prob=False,
                                     class_conditional=True,
                                     norm='l1',
                                     num_bins=num_bins,
                                     threshold=threshold)


def compute_all_metrics(labels, probs):
    """Computes all GCE metrics."""
    parameters = [['even', 'adaptive'], [True, False], [True, False],
                  [0.0, 0.01], ['l1', 'l2']]
    params = list(itertools.product(*parameters))
    measures = []
    for p in params:
        def metric(labels, probs, num_bins=30, p=p):
            """Implements Expected Calibration Error."""
            return general_calibration_error(labels,
                                             probs,
                                             binning_scheme=p[0],
                                             max_prob=p[1],
                                             class_conditional=p[2],
                                             threshold=p[3],
                                             norm=p[4],
                                             num_bins=num_bins)
        measures.append(metric(labels, probs))
    return np.array(measures)
