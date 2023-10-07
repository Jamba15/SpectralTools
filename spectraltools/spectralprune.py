import warnings

try:
    from tensorflow.keras.models import Model, model_from_json
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.backend import ones
    from tensorflow.keras.models import clone_model

except ModuleNotFoundError:
    from keras.models import Model, model_from_json
    from keras.layers import Dense
    from keras.backend import ones
    from keras.models import clone_model

import numpy as np
from .spectraldense import Spectral


def layers_unwrap(model: Model):
    """
    Return all the layers, including nested, in a TensorFlow FUNCTIONAL or SEQUENTIAL model.
    :param model: model to be analyzed
    :return: the list of all the layers
    """
    layers = []
    for lay in model.layers:
        try:
            layers += lay.layers
        except AttributeError:
            layers += [lay]
    return layers


def eigenvalue_cutoff(model: Model, perc):
    """
    Find the cutoff value between the eigenvalues according to percentile and update the global variable used to
    retrieve the spectral layers in the correct order in the future. It only takes into account the 'diag_end' eigenvalues.
    :param model: model in which eigenvalues cutoff will be calculated (excluded last layer)
    :param perc: percentile of eigenvalues (nodes) to be removed between every Spectral layer
    :return: the cutoff value and the spectral layers indexes in the model
    """
    index_list = []
    eigvals = []
    layers = layers_unwrap(model)

    # Cycle through all the layers. If a layer is spectral, save the eigenvalues and
    # the index of the layer in the model.
    for ind, lay in enumerate(layers):
        try:
            if isinstance(lay, Spectral):
                eigvals.append(lay.get_eigenvalues()['diag_end'].squeeze())
                index_list.append({"index": ind,
                                   "layer": type(lay).__name__})
        except AttributeError:
            pass

    eigvals = np.concatenate(eigvals, axis=0).flatten()

    # Calculate the cutoff value
    cut_off = np.percentile(abs(eigvals), np.clip(perc, 0, 100), 0)

    percent = np.sum(abs(eigvals) < cut_off) / len(eigvals) * 100
    print(f"Number of nodes masked: {np.sum(abs(eigvals) < cut_off)} out of {len(eigvals)} ({percent:.2f}%)")

    return cut_off, index_list


def prune_percentile(model: Model, percentile_threshold: float, compile_dictionary: dict):
    """
    Prune the model based on a percentile of eigenvalues between each Spectral layer.
    The function create a copy of the model and prune the copy. The original model is not modified.
    :param compile_dictionary: Dictionary containing the compilation parameters
    :param model: The neural network model to be pruned
    :param percentile_threshold: Percentile for pruning
    :return: Optionally returns the pruned model copy
    """
    if percentile_threshold < 0 or percentile_threshold > 100:
        raise ValueError("Percentile must be between 0 and 100.")

    cut_off, index_list = eigenvalue_cutoff(model, percentile_threshold)

    target_model = clone_model(model)
    target_model.set_weights(model.get_weights())

    layers = layers_unwrap(target_model)

    for indx in index_list:
        layers[indx['index']].mask_diag_end(cut_off)

    # Ricompila il modello
    target_model.compile(**compile_dictionary)

    return target_model


def metric_based_pruning(model: Model,
                         eval_dictionary: dict,
                         compile_dictionary: dict,
                         compare_metric='accuracy',
                         max_delta_percent=10,
                         **kwargs):
    """
    This function prunes the model based on the metric specified in the compare_metric parameter. The pruning is then
    iterated until the metric does not change more than max_delta_percent. The model is modified in place my masking
    the nodes with the lowest eigenvalues.
    :param model:
    `dict(x=x_train, y=y_train, batch_size=300, epochs=10, verbose=0)`
    :param compare_metric: Can be either 'loss' or 'accuracy'
    :param compile_dictionary: Dictionary containing the compilation parameters to be passed to the compile method
    :param max_delta_percent: Maximum delta in the metric before stopping the pruning
    :param eval_dictionary: Dictionary containing the evaluation parameters.
    :param kwargs: Optional arguments for the percentile_step (default 5)
    :return:
    """

    # Get the metric from the model
    if compare_metric not in model.metrics_names:
        raise ValueError(f"Metric {compare_metric} not found in model.metrics_names. "
                         f"Available metrics are {model.metrics_names}")
    # extract the corresponding index
    metrix_index = model.metrics_names.index(compare_metric)
    initial_metric_value = model.evaluate(**eval_dictionary)[metrix_index]

    # Prune the model with a for loop until the metric does not change more than max_delta_percent. The loop is on the
    # percentile of eigenvalues to be removed.
    for perc in range(0, 100, kwargs.get('percentile_step', 5)):
        pruned_model = prune_percentile(model, perc, compile_dictionary=compile_dictionary)
        new_metric_value = pruned_model.evaluate(**eval_dictionary, verbose=1)[metrix_index]
        current_delta = (abs(new_metric_value - initial_metric_value) / initial_metric_value) * 100
        print(
            f"Pruning with {perc}% of eigenvalues removed. Delta in {pruned_model.metrics_names[metrix_index]}: {current_delta:.4f}%")
        if current_delta > max_delta_percent:
            # restore the model to the iteration before
            if perc - 5 < 0:
                raise ValueError("The model has not be pruned_model, percentile is set to 0. Maybe the max_delta_percent is too low.")
            else:
                return prune_percentile(model, perc - 5, compile_dictionary=compile_dictionary)
    warnings.warn("The pruning has reached the maximum number of iterations. Maybe percentile_step is too low or "
                  "max_delta_percent is too high.")
    return None








