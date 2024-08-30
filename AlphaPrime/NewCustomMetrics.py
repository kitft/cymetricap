import tensorflow.keras as tfk
import tensorflow as tf


class LaplacianLoss(tfk.metrics.Metric):
    def __init__(self, name='laplacian_loss', **kwargs):
        super(LaplacianLoss, self).__init__(name=name, **kwargs)
        self.laplacian_loss = self.add_weight(name='lpl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['laplacian_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.laplacian_loss)/(self.count+1)
        self.laplacian_loss.assign_add(new_value)
        #tf.print("hiasg")
        #tf.print(tf.shape(self.laplacian_loss))
        self.count.assign_add(1)

    def result(self):
        return self.laplacian_loss

    def reset_state(self):
        self.laplacian_loss.assign(0)
        self.count.assign(0)

class LaplacianSpecialLoss(tfk.metrics.Metric):
    def __init__(self, name='laplacian_special_loss', **kwargs):
        super(LaplacianSpecialLoss, self).__init__(name=name, **kwargs)
        self.laplacian_loss = self.add_weight(name='lpls', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['laplacian_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.laplacian_loss)/(self.count+1)
        self.laplacian_loss.assign_add(new_value)
        #tf.print("hiasg")
        #tf.print(tf.shape(self.laplacian_loss))
        self.count.assign_add(1)

    def result(self):
        return self.laplacian_loss

    def reset_state(self):
        self.laplacian_loss.assign(0)
        self.count.assign(0)



def laplacian_measure_loss(model, validation_data):
    r"""Computes the Transition loss measure.

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), tf.float32]): NN input

    Returns:
        tf.float32: Transition loss measure
    """
    #X_val, aux = validation_data
    X_val = validation_data["X_val"]
    pullbacks = validation_data["val_pullbacks"]
    invmetrics = validation_data["inv_mets_val"]
    sources = validation_data["sources_val"]

    #X_val = tf.cast(X_val, tf.float32)
    #y_val = tf.cast(y_val, tf.float32)
    #aux=tf.cast(aux, tf.float32)
    #sort out this float32 problem!
    return tf.math.reduce_mean(
        model.compute_laplacian_loss(X_val,pullbacks,invmetrics,sources))



def laplacian_special_measure_loss(model, validation_data):
    r"""Computes the Transition loss measure.

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), tf.float32]): NN input

    Returns:
        tf.float32: Transition loss measure
    """
    X_val = validation_data["special_points_val"]
    pullbacks = validation_data["special_pullback_val"]
    invmetrics = validation_data["inv_mets_special_val"]
    sources = validation_data["sources_special_val"]

    #sort out this float32 problem!
    return tf.math.reduce_mean(
        model.compute_laplacian_loss(X_val, pullbacks, invmetrics, sources))

laplacian_measure_tf = tf.function(func=laplacian_measure_loss)
laplacian_special_measure_tf= tf.function(func=laplacian_special_measure_loss)

class LaplacianCallback(tfk.callbacks.Callback):
    """Callback that tracks the transition loss weighted over the CY."""
    def __init__(self, validation_data, initial=False):
        r"""A callback which computes the transition measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
        """
        super(LaplacianCallback, self).__init__()
        self.data=validation_data
        self.initial=initial
        
    def on_epoch_end(self, epoch, logs=None):
        r"""Computes transition measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        laplacian = laplacian_measure_tf(self.model, self.data)

        cb_res = laplacian.numpy().tolist()
        logs['laplacian_val'] = cb_res
        if cb_res <= 1e-3:
            print(' - Laplacian measure val: {:.4e}'.format(cb_res))
        else:
            print(' - Laplacian measure val: {:.4f}'.format(cb_res))
    
    def on_train_begin(self, logs=None):
        r"""Compute transition measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs)

class LaplacianSpecialCallback(tfk.callbacks.Callback):
    """Callback that tracks the special laplacian loss at a specific point."""
    def __init__(self, validation_data, initial=False):
        r"""A callback which computes the special laplacian measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
        """
        super(LaplacianSpecialCallback, self).__init__()
        self.data = validation_data
        self.initial = initial
        
    def on_epoch_end(self, epoch, logs=None):
        r"""Computes special laplacian measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        special_laplacian = laplacian_special_measure_tf(self.model, self.data)

        cb_res = special_laplacian.numpy().tolist()
        logs['special_laplacian_val'] = cb_res
        if cb_res <= 1e-3:
            print(' - Special Laplacian measure val: {:.4e}'.format(cb_res))
        else:
            print(' - Special Laplacian measure val: {:.4f}'.format(cb_res))
    
    def on_train_begin(self, logs=None):
        r"""Compute special laplacian measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs)