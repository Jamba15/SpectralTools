from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers as layer_type
import numpy as np
from .spectraldense import Spectral


class dense_assigner:
    def __init__(self,
                 new_shape,
                 new_kernel,
                 new_bias,
                 index=None):

        self.new_shape = new_shape
        self.new_kernel = new_kernel
        self.new_bias = new_bias
        self.index = index

    def assign(self, layer: Dense):
        layer.kernel.assign(self.new_kernel)
        if self.new_bias is not None:
            layer.bias.assign(self.new_bias)


class spectral_assigner:
    def __init__(self,
                 new_shape,
                 new_base,
                 new_diag_start,
                 new_diag_end,
                 new_bias,
                 index=None):

        self.new_shape = new_shape
        self.index = index
        self.new_base = new_base
        self.new_diag_start = new_diag_start
        self.new_diag_end = new_diag_end
        self.new_bias = new_bias

    def assign(self, layer: Spectral):
        layer.base.assign(self.new_base)
        layer.diag_start.assign(self.new_diag_start)
        layer.diag_end.assign(self.new_diag_end)

        if self.new_bias is not None:
            layer.bias.assign(self.new_bias)


def layers_unwrap(model: Model):
    """
    Return all the layers, including nested, in a model
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
    retrieve the spectral layers in the correct order in the future.
    :param model: model in which eigenvalues cutoff will be calculated (excluded last layer)
    :param perc: percentile of eigenvalues (nodes) to be removed between every Spectral layer
    :return: the cutoff value
    """
    index_list = []
    eigvals = []
    layers = layers_unwrap(model)

    for ind, lay in enumerate(layers):
        try:
            if lay.is_diag_end_trainable and (len(lay.outbound_nodes) != 0):
                eigvals.append(lay.return_diag())
        except AttributeError:
            pass

    for ind, lay in enumerate(layers):
        inbound, outbound = near_layer(lay)
        if type(inbound).__name__ == 'Spectral' or type(outbound).__name__ == 'Spectral' or type(lay).__name__ == 'Spectral':
            index_list.append({"index": ind,
                               "layer": type(lay).__name__})

    return np.percentile(abs(np.concatenate(eigvals, axis=0)), np.clip(perc, 0, 100), 0), index_list


def compare_eigenvalues(previous_layer: Spectral,
                        next_layer: Spectral,
                        cut_off: float,
                        mode: str):
    condition_start = previous_layer.conditions(cut_off)["diag_end"]
    condition_end = next_layer.conditions(cut_off)["diag_start"]

    if mode == 'or':
        return np.logical_or(condition_start, condition_end)

    elif mode == 'and':
        return np.logical_and(condition_start, condition_end)

    else:
        raise AttributeError('Insert a valid mode: or - and')


def near_layer(current_layer: layer_type):
    list_next = current_layer.outbound_nodes
    list_previous = current_layer.inbound_nodes
    inbound = None
    outbound = None
    if (len(list_next) > 1 or len(list_previous) > 1) and type(current_layer).__name__ == "Spectral":
        raise ValueError('Branched spectral non supported')
    else:
        if len(list_next) != 0 and list_next:
            outbound = list_next[0].outbound_layer

        if len(list_previous) != 0:
            inbound = list_previous[0].inbound_layers
    return inbound, outbound


def dense_weights_distiller(to_prune: Dense, cut_off: float):
    inbound, outbound = near_layer(current_layer=to_prune)

    if type(inbound).__name__ == "Spectral":
        in_cond = inbound.conditions(cut_off)["diag_end"]
    else:
        in_cond = np.ones(shape=to_prune.input_shape[1],
                          dtype=bool)

    if type(outbound).__name__ == "Spectral":
        out_cond = outbound.conditions(cut_off)["diag_start"]
    else:
        out_cond = np.ones(shape=to_prune.output_shape[1],
                           dtype=bool)

    new_kernel = to_prune.kernel.numpy()[:, out_cond]
    new_kernel = new_kernel[in_cond, :]

    if to_prune.bias is not None:
        new_bias = to_prune.bias.numpy()[out_cond]
    else:
        new_bias = None

    return dense_assigner(new_shape=np.sum(out_cond*1),
                          new_kernel=new_kernel,
                          new_bias=new_bias)


def spectral_weights_distiller(to_prune: Spectral, cut_off: float, link_mode: str = 'and'):
    inbound, outbound = near_layer(current_layer=to_prune)

    in_cond = to_prune.conditions(cut_off)["diag_start"]
    out_cond = to_prune.conditions(cut_off)["diag_end"]

    if type(inbound).__name__ == "Spectral":
        in_cond = compare_eigenvalues(previous_layer=inbound,
                                      next_layer=to_prune,
                                      cut_off=cut_off,
                                      mode=link_mode)

    if type(outbound).__name__ == "Spectral":
        out_cond = compare_eigenvalues(previous_layer=to_prune,
                                       next_layer=outbound,
                                       cut_off=cut_off,
                                       mode=link_mode)

    new_diag_start = to_prune.diag_start.numpy()[in_cond, :]
    new_diag_end = to_prune.diag_end.numpy()[:, out_cond]
    new_base = to_prune.base.numpy()[:, out_cond]
    new_base = new_base[in_cond, :]

    if to_prune.bias is not None:
        new_bias = to_prune.bias.numpy()[out_cond]
    else:
        new_bias = None

    return spectral_assigner(new_shape=np.sum(out_cond*1),
                             new_base=new_base,
                             new_diag_end=new_diag_end,
                             new_diag_start=new_diag_start,
                             new_bias=new_bias)


def spectral_pruning(model: Model, percentile: int):
    cut_off, index_list = eigenvalue_cutoff(model, percentile)
    new_weights = []
    layers = layers_unwrap(model)

    # Creating the new .json file
    for lay in index_list:
        if lay["layer"] == 'Spectral':
            new_assigner = spectral_weights_distiller(layers[lay['index']], cut_off)
        elif lay["layer"] == 'Dense':
            new_assigner = dense_weights_distiller(layers[lay['index']], cut_off)
        else:
            continue
        new_assigner.index = lay['index']
        new_weights.append(new_assigner)

        to_reshape = layers[new_assigner.index]
        to_reshape.units = new_assigner.new_shape

    new_json = model.to_json()

    # Creating the new smaller model

    custom_objects = {'Spectral': Spectral}
    new_model = model_from_json(new_json, custom_objects=custom_objects)
    new_layers = layers_unwrap(new_model)

    # Import pruned weights
    for assigner in new_weights:
        pruned = new_layers[assigner.index]
        assigner.assign(pruned)

    # Import all the others
    to_jump = [lay["index"] for lay in index_list]

    for n, lay in enumerate(new_layers):
        try:
            to_jump.index(n)
        except ValueError:
            lay.set_weights(layers[n].get_weights())

    new_model.compile(**model._get_compile_args())
    return new_model


def find_spectral(model):
    layers = layers_unwrap(model)
    index = []
    for ind, lay in enumerate(layers):
        if type(lay).__name__ == 'Spectral':
            index.append(ind)
    return index


def spectral_pretrain(model, fit_dictionary, eval_dictionary, max_drop, compare_with='acc'):
    spectral_layers = find_spectral(model)
    layers = layers_unwrap(model)
    for index in spectral_layers:
        current: Spectral = layers[index]
        current.is_base_trainable = False
        current.is_diag_end_trainable = True
        current.is_diag_start_trainable = False

    new_json = model.to_json()
    custom_objects = {'Spectral': Spectral}
    spec_only = model_from_json(new_json, custom_objects=custom_objects)
    spec_only.compile(**model._get_compile_args())
    spec_only.fit(**fit_dictionary)
    if compare_with == 'acc':
        index = 1
    else:
        index = 0

    acc = spec_only.evaluate(**eval_dictionary)[index]
    new_acc = acc
    p = 5

    while abs(new_acc - acc)/acc < (np.clip(max_drop, 0, 99)/100):
        new_model = spectral_pruning(spec_only, percentile=p)
        new_acc = new_model.evaluate(**eval_dictionary)[index]
        p += 5
        print(abs(new_acc - acc)/acc)

    return new_model
