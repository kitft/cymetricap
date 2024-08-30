from cymetric.models.fubinistudy import FSModel
from laplacian_funcs import *
import tensorflow as tf
import os
import numpy as np
from pympler import tracker

def point_vec_to_complex(p):
    #if len(p) == 0: 
    #    return tf.constant([[]])
    plen = (p.shape[-1])//2
    return tf.complex(p[..., :plen],p[..., plen:])

class GreenModel(FSModel):
    r"""FreeModel from which all other models inherit.

    The training and validation steps are implemented in this class. All
    other computational routines are inherited from:
    cymetric.models.fubinistudy.FSModel
    
    Example:
        Assume that `BASIS` and `data` have been generated with a point 
        generator.

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from cymetric.models.tfmodels import FreeModel
        >>> from cymetric.models.tfhelper import prepare_tf_basis
        >>> tfk = tf.keras
        >>> data = np.load('dataset.npz')
        >>> BASIS = prepare_tf_basis(np.load('basis.pickle', allow_pickle=True))
    
        set up the nn and FreeModel

        >>> nfold = 3
        >>> ncoords = data['X_train'].shape[1]
        >>> nn = tfk.Sequential(
        ...     [   
        ...         tfk.layers.Input(shape=(ncoords)),
        ...         tfk.layers.Dense(64, activation="gelu"),
        ...         tfk.layers.Dense(nfold**2),
        ...     ]
        ... )
        >>> model = FreeModel(nn, BASIS)

        next we can compile and train

        >>> from cymetric.models.metrics import TotalLoss
        >>> metrics = [TotalLoss()]
        >>> opt = tfk.optimizers.Adam()
        >>> model.compile(custom_metrics = metrics, optimizer = opt)
        >>> model.fit(data['X_train'], data['y_train'], epochs=1)

        For other custom metrics and callbacks to be tracked, check
        :py:mod:`cymetric.models.metrics` and
        :py:mod:`cymetric.models.callbacks`.
    """
    def __init__(self, tfmodel, BASIS, metricModel, special_point, final_matrix,alpha=None, **kwargs):
        r"""FreeModel is a tensorflow model predicting CY metrics. 
        
        The output is
            
            .. math:: g_{\text{out}} = g_{\text{NN}}
        
        a hermitian (nfold, nfold) tensor with each float directly predicted
        from the neural network.

        NOTE:
            * The model by default does not train against the ricci loss.
                
                To enable ricci training, set `self.learn_ricci = True`,
                **before** the tracing process. For validation data 
                `self.learn_ricci_val = True`,
                can be modified separately.

            * The models loss contributions are

                1. sigma_loss
                2. kaehler loss
                3. transition loss
                4. ricci loss (disabled)
                5. volk loss

            * The different losses are weighted with alpha.

            * The (FB-) norms for each loss are specified with the keyword-arg

                >>> model = FreeModel(nn, BASIS, norm = [1. for _ in range(5)])

            * Set kappa to the kappa value of your training data.

                >>> kappa = np.mean(data['y_train'][:,-2])

        Args:
            tfmodel (tfk.model): the underlying neural network.
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from cymetric.pointgen.pointgen.
            alpha ([5//NLOSS], float): Weighting of each loss contribution.
                Defaults to None, which corresponds to equal weights.
        """
        super(GreenModel, self).__init__(BASIS=BASIS, **kwargs)
        self.model = tfmodel
        self.NLOSS = 3
        # variable or constant or just tensor?
        if alpha is not None:
            self.alpha = [tf.Variable(a, dtype=tf.float32) for a in alpha]
        else:
            self.alpha = [tf.Variable(1., dtype=tf.float32) for _ in range(self.NLOSS)]
        self.learn_transition = tf.cast(True, dtype=tf.bool)
        self.learn_laplacian = tf.cast(True, dtype=tf.bool)
        self.learn_special_laplacian = tf.cast(True, dtype=tf.bool)

        #self.learn_kaehler = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        #self.learn_volk = tf.cast(False, dtype=tf.bool)

        self.custom_metrics = None
        #self.kappa = tf.cast(BASIS['KAPPA'], dtype=tf.float32)
        self.gclipping = float(5.0)
        # add to compile?
        #self.sigma_loss = sigma_loss(self.kappa, tf.cast(self.nfold, dtype=tf.float32))
        self.metricModel =metricModel
        self.sigmoid_for_nn = lambda x: sigmoid_like_function(x, transition_point=0.1, steepness=2)

        self.special_point=special_point
        self.special_pullback=tf.cast(self.pullbacks((tf.expand_dims(special_point,axis=0)))[0],dtype=tf.complex64)#self.pullbacks takes real arguments
        self.special_metric=self.metricModel(tf.expand_dims(special_point,axis=0))[0]
        self.final_matrix=final_matrix
        # Check if special_metric is the pullback of final_matrix

        self.kahler_t = tf.math.real(self.BASIS['KMODULI'][0])
        self.geodesic_distance_vec_function= lambda cpoints: vectorized_geodesic_distance_CPn(
            point_vec_to_complex(self.special_point),
            cpoints,
            kahler_t=self.kahler_t,
            metricijbar=self.final_matrix
        )
        self.test_pulled_back_matrix()
        self.dim_output=self.model.dim_output

    def test_pulled_back_matrix(self):
        actual_metric_from_matrix=compute_Gijbar_from_Hijbar(self.final_matrix,point_vec_to_complex(self.special_point),self.kahler_t)
        pulled_back_matrix = tf.einsum('ai,BJ,iJ->aB', 
                                       self.special_pullback,tf.math.conj(self.special_pullback), 
                                       actual_metric_from_matrix)
        
        # Compare the pulled back matrix with special_metric
        is_equal = tf.reduce_all(tf.math.abs(pulled_back_matrix - self.special_metric) < 1e-6)
        tf.print("Is special_metric the pullback of final_matrix?", is_equal)
        
        if not is_equal:
            tf.print("Warning: special_metric is not the pullback of final_matrix")
            tf.print("Pulled back matrix:", pulled_back_matrix)
            tf.print("Special metric:", self.special_metric)



    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, self.nProjective))
        current_patch_mask = self._indices_to_mask(patch_indices)
        fixed = self._find_max_dQ_coords(points)
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        if self.nhyper == 1:
            other_patches = tf.gather(self.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = self._generate_patches_vec(combined)
        
        other_patches = tf.reshape(other_patches, (-1, self.nProjective))
        other_patch_mask = self._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, self.nTransitions, axis=-2)#expanded points
        patch_points = self._get_patch_coordinates(exp_points, tf.cast(other_patch_mask, dtype=tf.bool)) # other patches
        real_patch_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        gj = self.model(real_patch_points, training=True)
        gi = tf.repeat(self.model(points), self.nTransitions, axis=0)
        all_t_loss = tf.math.abs(gi-gj)
        all_t_loss = tf.reshape(all_t_loss, (-1, self.nTransitions,self.dim_output))
        all_t_loss = tf.math.reduce_sum(all_t_loss**self.n[1], axis=(-2,-1))
        
        return all_t_loss/(self.nTransitions)


    def compute_laplacian_loss(self,x,pullbacks,invmetrics,sources):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        #cast to real
        #-2*laplacian because this is the actual 'laplace-beltrami operator', not just gabbar del delbar
        lpl_losses=tf.math.abs(tf.math.real(-2*laplacian(self,x,pullbacks,invmetrics))-sources)
        all_lpl_loss = lpl_losses**self.n[0]
        return all_lpl_loss

    def HYM_measure_val(self,greenmodel,datagreen):
        #arguments: betamodel, databeta
        #outputs: weighted by the point weights, the failure to solve the equation i.e.:
        # 1: number: sum(w*|laplacian(beta)-rho|)/|sum(w.|rho|)|, where w is the point weight, rho is the source
        # 2: vector: w*|laplacian(beta)-rho|/|sum(w.|rho|)|, where w is the point weight, rho is the source
        # 3: number: w*|laplacian(beta)-rho|)/sum(w.|rho|), where w is the point weight, rho is the source
    
        vals=datagreen['y_val'][:,0]*tf.math.abs(-2*laplacian(greenmodel,datagreen['X_val'],datagreen['val_pullbacks'],datagreen['inv_mets_val'])-datagreen['sources_val'])
        val=tf.reduce_mean(vals, axis=-1)
        absolutevalsofsourcetimesweight=datagreen['y_val'][:,0]*tf.math.abs(datagreen['sources_val'])
        mean_ofabsolute_valofsourcetimesweight=tf.reduce_mean(absolutevalsofsourcetimesweight, axis=-1)

        return val/mean_ofabsolute_valofsourcetimesweight, vals/mean_ofabsolute_valofsourcetimesweight,vals/absolutevalsofsourcetimesweight


    def call(self, input_tensor, j_elim=None):
        r"""Prediction of the model.

        .. math::

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Prediction at each point.
        """
        # nn prediction
        cpoints=point_vec_to_complex(input_tensor)
        geodesic_distance= self.geodesic_distance_vec_function(cpoints)
        local_model=self.local_model_of_greens_function(geodesic_distance)

        to_multiply_nn_global=self.sigmoid_for_nn(geodesic_distance)
        to_multiply_nn_local=1-to_multiply_nn_global
        to_dot_to_nn=tf.stack([to_multiply_nn_global,to_multiply_nn_local],axis=-1)
        nn_prediction=self.model(input_tensor)#take 0th so its shape is a pure vector.
        nn_prediction_out=tf.einsum('...i,...i->...',to_dot_to_nn,nn_prediction)

        return local_model+nn_prediction_out
        

    def local_model_of_greens_function(self,geodesic_distance):
        area_of_unit_sphere_in_2ndim=2*np.pi**(float(self.nfold))/tf.math.exp(tf.math.lgamma(float(self.nfold)))
        if int(2*self.nfold)!=2:
            c_n = 1/((2*self.nfold-2)*area_of_unit_sphere_in_2ndim)# positive sign for 2n>2
            return c_n*geodesic_distance**(-(2*self.nfold-2))
        elif int(2*self.nfold)==2:
            c_n = -1/(2*np.pi)# negative sign for 2n==2
            return c_n*np.log(geodesic_distance)


    def compile(self, custom_metrics=None, **kwargs):
        r"""Compiles the model.
        kwargs takes any argument of regular `tf.model.compile()`
        Example:
            >>> model = FreeModel(nn, BASIS)
            >>> from cymetric.models.metrics import TotalLoss
            >>> metrics = [TotalLoss()]
            >>> opt = tfk.optimizers.Adam()
            >>> model.compile(metrics=metrics, optimizer=opt)
        Args:
            custom_metrics (list, optional): List of custom metrics.
                See also :py:mod:`cymetric.models.metrics`. If None, no metrics
                are tracked during training. Defaults to None.
        """
        if custom_metrics is not None:
            kwargs['metrics'] = custom_metrics
        super(GreenModel, self).compile(**kwargs)

    @property
    def metrics(self):
        r"""Returns the model's metrics, including custom metrics.
        Returns:
            list: metrics
        """
        return self._metrics

    def train_step(self, data):
        r"""Train step of a single batch in model.fit().

        NOTE:
            1. The first epoch will take additional time, due to tracing.
            
            2. Warnings are plentiful. Disable on your own risk with 

                >>> tf.get_logger().setLevel('ERROR')
            
            3. The conditionals need to be set before tracing. 
            
            4. We employ under the hood gradient clipping.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # if len(data) == 3:
        #     x, sample_weight = data
        # else:
        #     sample_weight = None
        #     x, aux = dataX_train, y_train, train_pullback,inv_mets_train,sources_train
        y = None
        y_pred=None
        # print("hi")
        # print(x.shape)
        # print(len(x))
        # The 'y_train/val' arrays contain the integration weights and $\\Omega \\wedge \\bar\\Omega$ for each point. In principle, they can be used for any relevant pointwise information that could be needed during the training process."

        sample_weight = None
        x = data["X_train"]
        pbs = data["train_pullbacks"]
        invmets = data["inv_mets_train"]
        sources = data["sources_train"]
        x_special = data["special_points_train"]
        pbs_special = data["special_pullbacks_train"]
        invmets_special = data["inv_mets_special_train"]
        sources_special = data["sources_special_train"]
        with tf.GradientTape(persistent=False) as tape:
            trainable_vars = self.model.trainable_variables
            #tape.watch(trainable_vars)
            #automatically watch trainable vars
            # add other loss contributions.
            if self.learn_transition:
                t_loss =self.compute_transition_loss(x)
            else:
                t_loss = tf.zeros_like(x[:, 0])
            if self.learn_laplacian:
                lpl_loss = self.compute_laplacian_loss(x,pbs,invmets,sources)
                #print("lpl beta")
            else:
                lpl_loss = tf.zeros_like(x[:, 0])
            if self.learn_special_laplacian:
                lpl_special_loss = self.compute_laplacian_loss(x_special,pbs_special,invmets_special,sources_special)
                #print("lpl beta")
            else:
                lpl_special_loss = tf.zeros_like(x_special[:, 0])

            #omega = tf.expand_dims(y[:, -1], -1)
            #sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
            total_loss = self.alpha[0]*lpl_loss +\
                self.alpha[1]*t_loss +\
                self.alpha[2]*lpl_special_loss
            # weight the loss.
            if sample_weight is not None:
                total_loss *= sample_weight
            total_loss_mean=tf.reduce_mean(total_loss)
        # Compute gradients
        gradients = tape.gradient(total_loss_mean, trainable_vars)
        # remove nans and gradient clipping from transition loss.
        gradients = [tf.where(tf.math.is_nan(g), 1e-8, g) for g in gradients]
        gradients, _ = tf.clip_by_global_norm(gradients, self.gclipping)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return metrics. NOTE: This interacts badly with any regular MSE
        # compiled loss. Make it so that only custom metrics are updated?
        loss_dict = {m.name: m.result() for m in self.metrics}
        loss_dict['loss'] = tf.reduce_mean(total_loss)
        loss_dict['laplacian_loss'] = tf.reduce_mean(lpl_loss)
        loss_dict['transition_loss'] = tf.reduce_mean(t_loss)
        loss_dict['special_laplacian_loss'] = tf.reduce_mean(lpl_special_loss)
        return loss_dict

    def test_step(self, data):
        r"""Same as train_step without the outer gradient tape.
        Does *not* update the NN weights.

        NOTE:
            1. Computes the exaxt same losses as train_step
            
            2. Ricci loss val can be separately enabled with
                
                >>> model.learn_ricci_val = True
            
            3. Requires additional tracing.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # unpack data
        # if len(data) == 3:
        #     x, aux, sample_weight = data
        # else:
        #     sample_weight = None
        #     x, aux = data
        #x,sample_weight, pbs, invmets, sources = data.values()
        y = None
        y_pred=None
        x = data["X_val"]
        sample_weight = None
        pbs = data["val_pullbacks"]
        invmets = data["inv_mets_val"]
        sources = data["sources_val"]
        x_special = data["special_points_val"]
        pbs_special = data["special_pullbacks_val"]
        invmets_special = data["inv_mets_special_val"]
        sources_special = data["sources_special_val"]

        if self.learn_transition:
            t_loss = self.compute_transition_loss(x)
        else:
            t_loss = tf.zeros_like(x[:, 0])
        if self.learn_laplacian:
            lpl_loss = self.compute_laplacian_loss(x,pbs,invmets,sources)
        else:
            lpl_loss = tf.zeros_like(x[:, 0])
        if self.learn_special_laplacian:
            lpl_special_loss = self.compute_laplacian_loss(x_special,pbs_special,invmets_special,sources_special)
        else:
            lpl_special_loss = tf.zeros_like(x_special[:, 0])

        #omega = tf.expand_dims(y[:, -1], -1)
        #sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
        total_loss = self.alpha[0]*lpl_loss +\
            self.alpha[1]*t_loss+\
            self.alpha[2]*lpl_special_loss
        # weight the loss.
        if sample_weight is not None:
            total_loss *= sample_weight
        loss_dict = {m.name: m.result() for m in self.metrics}
        loss_dict['loss'] = tf.reduce_mean(total_loss)
        loss_dict['laplacian_loss'] = tf.reduce_mean(lpl_loss)
        loss_dict['transition_loss'] = tf.reduce_mean(t_loss)
        loss_dict['special_laplacian_loss'] = tf.reduce_mean(lpl_special_loss)
        return loss_dict


    def save(self, filepath, **kwargs):
        r"""Saves the underlying neural network to filepath.

        NOTE: 
            Currently does not save the whole custom model.

        Args:
            filepath (str): filepath
        """
        # TODO: save graph? What about Optimizer?
        # https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        self.model.save(filepath=filepath, **kwargs)


def prepare_dataset_Green(point_gen, data, dirname, special_point,metricModel,BASIS,val_split=0.1, ltails=0, rtails=0, normalize_to_vol_j=True,batch_size=100,max_tolerance_for_gradient_descent=1e-7):
    r"""Prepares training and validation data from point_gen.

    Note:
        The dataset will be saved in `dirname/dataset.npz`.

    Args:
        point_gen (PointGenerator): Any point generator.
        n_p (int): # of points.
        dirname (str): dir name to save data.
        val_split (float, optional): train-val split. Defaults to 0.1.
        ltails (float, optional): Discarded % on the left tail of weight 
            distribution.
        rtails (float, optional): Discarded % on the left tail of weight 
            distribution.
        normalize_to_vol_j (bool, optional): Normalize such that

            .. math::
            
                \int_X \det(g) = \sum_p \det(g) * w|_p  = d^{ijk} t_i t_j t_k

            Defaults to True.

    Returns:
        np.float: kappa = vol_k / vol_cy
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    n_p = len(data['X_train']) + len(data['X_val'])
    new_np = int(round(n_p/(1-ltails-rtails)))
    pwo = point_gen.generate_point_weights(new_np, omega=True)
    if len(pwo) < new_np:
        new_np = int((new_np-len(pwo))/len(pwo)*new_np + 100)
        pwo2 = point_gen.generate_point_weights(new_np, omega=True)
        pwo = np.concatenate([pwo, pwo2], axis=0)
    new_np = len(pwo)
    sorted_weights = np.sort(pwo['weight'])
    lower_bound = sorted_weights[round(ltails*new_np)]
    upper_bound = sorted_weights[round((1-rtails)*new_np)-1]
    mask = np.logical_and(pwo['weight'] >= lower_bound,
                          pwo['weight'] <= upper_bound)
    weights = np.expand_dims(pwo['weight'][mask], -1)
    omega = np.expand_dims(pwo['omega'][mask], -1)
    omega = np.real(omega * np.conj(omega))
    
    points= pwo['point'][mask]
    points = tf.cast(points,tf.complex64)

    t_i = int((1-val_split)*new_np)
    

    if normalize_to_vol_j:
        pbs = point_gen.pullbacks(points)
        fs_ref = point_gen.fubini_study_metrics(points, vol_js=np.ones_like(point_gen.kmoduli))
        fs_ref_pb = tf.einsum('xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
        aux_weights = omega.flatten() / weights.flatten()
        norm_fac = point_gen.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / aux_weights)
        #print("point_gen.vol_j_norm")
        #print(point_gen.vol_j_norm)
        weights = norm_fac * weights # I.E. this is vol_j_norm/ integral of g_FS. That is, we normalise our volume to d_rst 1 1 1, when it is calculated with integral of omega wedge omegabar, i.e. just the weights. I.e. sum over just weights is that.
        # not sure if the above explanation is correct

    X_train=point_vec_to_real(points[:t_i])
    y_train = np.concatenate((weights[:t_i], omega[:t_i]), axis=1)
    X_val=point_vec_to_real(points[t_i:])
    y_val = np.concatenate((weights[t_i:], omega[t_i:]), axis=1)

    
    
    realpoints=tf.concat((tf.math.real(points), tf.math.imag(points)), axis=-1)
    realpoints=tf.cast(realpoints,tf.float32)

    X_train=tf.cast(X_train,tf.float32)
    y_train=tf.cast(y_train,tf.float32)
    X_val=tf.cast(X_val,tf.float32)
    y_val=tf.cast(y_val,tf.float32)
    #realpoints=tf.cast(realpoints,tf.float32)

    #X_train=tf.cast(data['X_train'],tf.float32)
    #y_train=tf.cast(data['y_train'],tf.float32)
    #X_val=tf.cast(data['X_val'],tf.float32)
    #y_val=tf.cast(data['y_val'],tf.float32)
    ncoords=int(len(X_train[0])/2)

    #y_train=data['y_train']
    #y_val=data['y_val']
    ys=tf.concat((y_train,y_val),axis=0)
    weights=tf.cast(tf.expand_dims(ys[:,0],axis=-1),tf.float32)
    omega=tf.cast(tf.expand_dims(ys[:,1],axis=-1),tf.float32)
    
    realpoints=tf.concat((X_train,X_val),axis=0)
    points=tf.complex(realpoints[:,0:ncoords],realpoints[:,ncoords:])

    new_np = len(realpoints)
    t_i = int((1-val_split)*new_np)


    mets = metricModel(realpoints)
    absdets = tf.abs(tf.linalg.det(mets))
    inv_mets=tf.linalg.inv(mets)
    inv_mets_train=inv_mets[:t_i]
    inv_mets_val=inv_mets[t_i:]

    #flat_weights=weights[:,0]*omega[:,0]**(-1)*1/6
    #cy_weights=flat_weights*absdets
    #print(tf.shape(weights))
    #print(tf.shape(flat_weights))
    ##print(tf.shape(absdets))
    #print(tf.shape(cy_weights))



    #still need to generate pullbacks apparently
    pullbacks = point_gen.pullbacks(points)
    train_pullbacks=tf.cast(pullbacks[:t_i],tf.complex64) 
    val_pullbacks=tf.cast(pullbacks[t_i:],tf.complex64) 

    # points = pwo['point'][mask]
    det = tf.math.real(absdets)  # * factorial / (2**nfold)
    #print("hi")
    det_over_omega = det / omega[:,0]
    #print("hi")
    volume_cy = tf.math.reduce_mean(weights[:,0], axis=-1)# according to raw CY omega calculation and sampling...
    #print("hi")
    vol_k_no6 = tf.math.reduce_mean(det_over_omega * weights[:,0], axis=-1)#missing factor of 6
    #print("hi")
    kappaover6 = tf.cast(vol_k_no6,tf.float32) / tf.cast(volume_cy,tf.float32)
    #rint(ratio)
    #print("hi")
    #tf.cast(kappaover6,tf.float32)
    #print("hi")
    det = tf.cast(det,tf.float32)
    # print('kappa over nfold factorial! ')
    # print(kappaover6) 


    kahler_t=tf.math.real(BASIS['KMODULI'][0])

    special_point_complex=point_vec_to_complex(special_point)
    special_pullback=tf.cast(point_gen.pullbacks(tf.expand_dims(special_point_complex,axis=0))[0],tf.complex64)

    nfold = tf.shape(special_pullback)[0].numpy()
    volume_for_sources = tf.cast(vol_k_no6 / np.math.factorial(nfold),tf.float32)
    print('Volume for sources: ', volume_for_sources)


    final_matrix = optimize_and_get_final_matrix(special_pullback, special_point, metricModel, kahler_t=kahler_t, plot_losses=False)
    radius=0.05
    min_radius=0.005
    num_points=len(ys)

    points_around_special=get_points_around_special(special_point_complex,radius,num_points,point_gen,uniform_on_radius=True,min_radius=min_radius,final_matrix=final_matrix,kahler_t=kahler_t,max_tolerance_for_gradient_descent=max_tolerance_for_gradient_descent)
    points_around_special=point_vec_to_real(points_around_special)
    special_points_train=points_around_special[0:t_i]
    special_points_val=points_around_special[t_i:]
    special_pullbacks_train=tf.cast(point_gen.pullbacks(point_vec_to_complex(special_points_train)),tf.complex64)
    special_pullbacks_val=tf.cast(point_gen.pullbacks(point_vec_to_complex(special_points_val)),tf.complex64)
    inv_mets_special_train=tf.cast(tf.linalg.inv(metricModel(special_points_train)),tf.complex64)# this cast is extraneous
    inv_mets_special_val=tf.cast(tf.linalg.inv(metricModel(special_points_val)),tf.complex64)# this cast is extraneous

    
    
    sources_train = -1 * (1/volume_for_sources) * tf.ones_like(y_train[:, 0])
    sources_val = -1 * (1/volume_for_sources) * tf.ones_like(y_val[:, 0])
    sources_special_train = -1 * (1/volume_for_sources) * tf.ones_like(special_points_train[:, 0])
    sources_special_val = -1 * (1/volume_for_sources) * tf.ones_like(special_points_val[:, 0])

 
    np.savez_compressed(os.path.join(dirname, 'dataset'),
                        X_train=X_train,
                        y_train=y_train,
                        train_pullbacks=train_pullbacks,
                        inv_mets_train=inv_mets_train,
                        sources_train=sources_train,
                        special_points_train=special_points_train,
                        special_pullbacks_train=special_pullbacks_train,
                        inv_mets_special_train=inv_mets_special_train,
                        sources_special_train=sources_special_train,
                        final_matrix=final_matrix,
                        X_val=X_val,
                        y_val=y_val,
                        val_pullbacks=val_pullbacks,
                        inv_mets_val=inv_mets_val,
                        sources_val=sources_val,
                        special_points_val=special_points_val,
                        special_pullbacks_val=special_pullbacks_val,
                        inv_mets_special_val=inv_mets_special_val,
                        sources_special_val=sources_special_val,
                        final_matrix_copy=final_matrix,
                        )
    print("print 'kappa/6'")
    return kappaover6#point_gen.compute_kappa(points, weights, omega)


def train_modelgreen(greenmodel, data_train, optimizer=None, epochs=50, batch_sizes=[64, 10000],
                verbose=1, custom_metrics=[], callbacks=[], sw=False):
    r"""Training loop for fixing the Kähler class. It consists of two 
    optimisation steps. 
        1. With a small batch size and volk loss disabled.
        2. With only MA and volk loss enabled and a large batchsize such that 
            the MC integral is a reasonable approximation and we don't lose 
            the MA progress from the first step.
    Args:
        greenmodel (cymetric.models.tfmodels): Any of the custom metric models.
        data (dict): numpy dictionary with keys 'X_train' and 'y_train'.
        optimizer (tfk.optimiser, optional): Any tf optimizer. Defaults to None.
            If None Adam is used with default hyperparameters.
        epochs (int, optional): # of training epochs. Every training sample will
            be iterated over twice per Epoch. Defaults to 50.
        batch_sizes (list, optional): batch sizes. Defaults to [64, 10000].
        verbose (int, optional): If > 0 prints epochs. Defaults to 1.
        custom_metrics (list, optional): List of tf metrics. Defaults to [].
        callbacks (list, optional): List of tf callbacks. Defaults to [].
        sw (bool, optional): If True, use integration weights as sample weights.
            Defaults to False.

    Returns:
        model, training_history
    """
    training_history = {}
    hist1 = {}
    hist2 = {}
    if sw:
        sample_weights = data_train['y_train'][:, -2]
    else:
        sample_weights = None
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    permint= tracker.SummaryTracker()
    for epoch in range(epochs):
        #print("internal")
        #print(permint.print_diff())
        batch_size = batch_sizes[0]
        greenmodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        if verbose > 0:
            print("\nEpoch {:2d}/{:d}".format(epoch + 1, epochs))
        history = greenmodel.fit(
            data_train,
            epochs=1, batch_size=batch_size, verbose=verbose,
            callbacks=callbacks, sample_weight=sample_weights
        )
        #print(history)
        for k in history.history.keys():
            if k not in hist1.keys():
                hist1[k] = history.history[k]
            else:
                hist1[k] += history.history[k]

        #add learning rate schedule
        # if loss is not decreasing, reduce learning rate

        if epoch>5 and hist1['laplacian_loss'][-1]>hist1['laplacian_loss'][-2]:
            greenmodel.optimizer.lr = greenmodel.optimizer.lr*0.9
            print("cutting LR, multiplying by 0.9 - new LR: " + str(greenmodel.optimizer.lr))

        #after 30 epochs, decrease learning rate by factor of 2
        if epoch==10:
            greenmodel.optimizer.lr = greenmodel.optimizer.lr*0.5
            print("cutting LR, multiplying by 0.1 - new LR: " + str(greenmodel.optimizer.lr))

        # after 60 epochs, decrease lr by a factor of 2
        if epoch==20:
            greenmodel.optimizer.lr = greenmodel.optimizer.lr*0.5
            print("cutting LR, multiplying by 0.1 - new LR: " + str(greenmodel.optimizer.lr))

        #after 90 epochs, decrease lr by a factor of 5
        if epoch==30:
            greenmodel.optimizer.lr = greenmodel.optimizer.lr*0.5
            print("cutting LR, multiplying by 0.1 - new LR: " + str(greenmodel.optimizer.lr))

        if tf.math.is_nan(hist1['loss'][-1]):
            break

    for k in set(list(hist1.keys())):
        #training_history[k] = hist2[k] if k in hist2 and max(hist2[k]) != 0 else hist1[k]
        training_history[k] = hist1[k]
    training_history['epochs'] = list(range(epochs))
    return greenmodel, training_history



# def compute_batched_func(compute_Q,input_vector,batch_size,weights):
#     #returns the Q vector, and the sqrt(g) Q vector
#     print("computing batched with batch size " + str(batch_size) + " and total length " + str(len(input_vector)))
#     resultall2=[]
#     for i in range(0, len(input_vector), batch_size):
#         batch = input_vector[i:i+batch_size]
#         if len(batch)<batch_size:
#             #copy batch as many times as necessary until you get batch_size
#             batch=tf.concat([batch for _ in range((batch_size//len(batch))+1)],axis=0)[0:batch_size]
            
#         result=compute_Q(batch)
#         resultall2.append(result)
#         result_temp=tf.math.real(tf.concat(resultall2,axis=0))
#         #fix incorrect length in final batch
#         length=min(len(input_vector),len(result_temp))
#         euler_all=weights[0:length]*result_temp[0:length]
#         euler=tf.reduce_mean(euler_all)
#         vol=tf.reduce_mean(weights[0:length])
#         print("in " + str(i+batch_size) + " euler: " + str(euler.numpy())+  " vol " + str(vol.numpy()))
#     #concatenate and fix length issue, also cast to real as it should be/is real
#     resultarr2=tf.math.real(tf.concat(resultall2,axis=0)[:len(input_vector)])
#     return resultarr2, euler_all


def compute_batched_func(compute_Q, input_vector, batch_size, weights):
    total_length = tf.shape(input_vector)[0]
    num_batches = (total_length + batch_size - 1) // batch_size

    result_array = tf.TensorArray(tf.float32, size=num_batches)
    euler_sum = tf.constant(0.0)
    weight_sum = tf.constant(0.0)

    for i in tf.range(num_batches):
        start = i * batch_size
        end = tf.minimum((i + 1) * batch_size, total_length)
        batch = input_vector[start:end]

        current_batch_size = tf.shape(batch)[0]
        #fix incorrect length in final batch
        if current_batch_size < batch_size:
            repeat_times = tf.cast(tf.math.ceil(batch_size / current_batch_size), tf.int32)
            batch=tf.tile(batch, [repeat_times,1])[0:batch_size]

        result = tf.math.real(compute_Q(batch))
        result_array = result_array.write(i, result)

        batch_weights = weights[start:end]
        number_of_points=tf.shape(batch_weights)[0]#can be different to batch_size on laast iteration
        batch_result = tf.math.real(result[:number_of_points])
        batch_euler = tf.reduce_sum(batch_weights * batch_result)
        batch_weight_sum = tf.reduce_sum(batch_weights)

        euler_sum += batch_euler
        weight_sum += batch_weight_sum

        tf.print("in", end, "euler:", euler_sum / float(min((i+1)*batch_size,total_length)), "vol", weight_sum/float(min((i+1)*batch_size,total_length)))

    #stack, concatenate and fix length issue, also cast to real as it should be/is real
    resultarr2 = tf.math.real(result_array.stack())
    resultarr2=tf.reshape(resultarr2,[-1])[:total_length]


    euler_all = weights[:total_length] * resultarr2

    return resultarr2, euler_all



def vector_to_hermitian_matrix(vec, n):
    """
    Construct a Hermitian matrix from a vector of real numbers.
    
    Args:
    vec (tf.Tensor): Vector of real numbers.
    n (int): Dimension of the square matrix.
    
    Returns:
    tf.Tensor: Hermitian matrix.
    """
    # Reshape the vector into a lower triangular matrix
    tril = tf.zeros((n, n), dtype=tf.complex64)
    tril = tf.linalg.set_diag(tril, tf.cast(vec[:n]/2, tf.complex64))
    
    # Calculate the number of elements in the lower triangular part (excluding diagonal)
    num_lower_tri = n * (n - 1) // 2
    
    # Split the remaining elements of vec into real and imaginary parts
    real_part = vec[n:n+num_lower_tri]
    imag_part = vec[n+num_lower_tri:]
    
    # Combine real and imaginary parts into complex numbers
    complex_elements = tf.complex(real_part, imag_part)
    
    # Create indices for the upper and lower triangular parts (excluding diagonal)
    upper_indices = tf.where(tf.linalg.band_part(tf.ones((n,n)), 0, -1) - tf.eye(n))
    
    # Update the lower triangular matrix with complex elements
    tril = tf.tensor_scatter_nd_update(tril, upper_indices, complex_elements)
    
    # Construct the Hermitian matrix
    return (tril + tf.math.conj(tf.transpose(tril)))



@tf.function
def sigmoid_like_function(x, transition_point=1, steepness=1):
    """
    Maps a number from [0, infty) to [0, 1].

    Args:
    x: Input value
    transition_point: Point where the function starts to flatten out
    steepness: Controls how quickly the function transitions

    Returns:
    A value between 0 and 1
    """
    x = tf.clip_by_value(x, 0, tf.float32.max)  # Ensure x is non-negative
    quadratic_part = (x / transition_point) ** 2
    return 1 - tf.exp(-steepness * quadratic_part)


def geodesic_distance_CPn(cpoint1, cpoint2, kahler_t, metricijbar=None):
    # Calculate geodesic distance between two points in CP^n
    # Check if point1 and point2 are complex-valued
    if not isinstance(cpoint1, tf.Tensor) or not isinstance(cpoint2, tf.Tensor) or cpoint1.dtype not in (tf.complex64, tf.complex128) or cpoint2.dtype not in (tf.complex64, tf.complex128):
        raise ValueError("Both point1 and point2 must be complex-valued tensors.")
    
    # Optionally, you can also check the dtype explicitly
    if cpoint1.dtype != tf.complex64 and cpoint1.dtype != tf.complex128:
        raise TypeError(f"Expected cpoint1 to be complex64 or complex128, but got {cpoint1.dtype}")
    if cpoint2.dtype != tf.complex64 and cpoint2.dtype != tf.complex128:
        raise TypeError(f"Expected cpoint2 to be complex64 or complex128, but got {cpoint2.dtype}")
    # Normalize the points using the metric
    if metricijbar is None:
        # Use identity metric for normalization
        p1 = cpoint1 / tf.cast(tf.sqrt(tf.math.real(tf.einsum('i,i->', cpoint1, tf.math.conj(cpoint1)))), tf.complex64)
        p2 = cpoint2 / tf.cast(tf.sqrt(tf.math.real(tf.einsum('i,i->', cpoint2, tf.math.conj(cpoint2)))), tf.complex64)
        
        # Calculate inner product with identity metric
        inner_product = tf.math.abs(tf.einsum('i,i->', p1, tf.math.conj(p2)))
    else:
        # Use specified metric for normalization
        p1 = cpoint1 / tf.cast(tf.sqrt(tf.math.real(tf.einsum('i,ij,j->', cpoint1, metricijbar, tf.math.conj(cpoint1)))), tf.complex64)
        p2 = cpoint2 / tf.cast(tf.sqrt(tf.math.real(tf.einsum('i,ij,j->', cpoint2, metricijbar, tf.math.conj(cpoint2)))), tf.complex64)
        # Calculate inner product with specified metric
        inner_product = tf.math.abs(tf.einsum('i,ij,j->', p1, metricijbar, tf.math.conj(p2)))
    
    # Clamp the inner product to [-1, 1] to avoid numerical issues
    inner_product = tf.clip_by_value(inner_product, -1.0, 1.0)
    
    # Calculate the geodesic distance
    distance = kahler_t * tf.math.acos(inner_product)
    
    return distance

@tf.function
def vectorized_geodesic_distance_CPn(special_point, cpoints, kahler_t=1.0, metricijbar=None):
    # Vectorized version of geodesic_distance_CPn
    return tf.vectorized_map(
        lambda p: geodesic_distance_CPn(special_point, p, kahler_t, metricijbar=metricijbar),
        cpoints
    )

def get_points_around_special(special_point_complex,radius,num_points,pg,uniform_on_radius=False,min_radius=0.01,final_matrix=None,kahler_t=1.0,max_tolerance_for_gradient_descent=1e-7):
    num_points_to_generate=num_points*2

    def poly(cpoints,pg):
        p_exp = tf.expand_dims(cpoints, 1)
        polys = tf.math.pow(p_exp, tf.cast(pg.BASIS['QB0'], tf.complex64))
        polys = tf.reduce_prod(polys, axis=-1)
        polys = tf.multiply(tf.cast(pg.BASIS['QF0'], tf.complex64), polys)
        polys = tf.reduce_sum(polys, axis=-1)
        return polys

    def poly_normed_abs_val(cpoints,pg):
        deg = tf.cast(tf.reduce_sum(tf.cast(pg.BASIS['QB0'], tf.complex64),axis=-1)[0],tf.complex64)
        polys=poly(cpoints,pg)
        polysnormed=polys*tf.einsum('xi,xi->x',tf.math.conj(cpoints),cpoints)**(-deg/2)
        return tf.math.abs(polysnormed)

    def generate_uniform_ball_c(center, radius, num_points, uniform_on_radius=False):
        """Generate points uniformly distributed in a complex ball around a center point."""
        """eps is the minimum value for the radius, so that we don't divide by zero eventually"""
        diminteger=tf.shape(center)[-1]
        dim = tf.cast(diminteger,tf.float32)  # Complex dimension

        # Generate random complex directions
        directions_real = tf.random.normal((num_points, diminteger))
        directions_imag = tf.random.normal((num_points, diminteger))
        directions = tf.complex(directions_real, directions_imag)
        directions = directions / tf.norm(directions, axis=1, keepdims=True)

        # Generate random radii for uniform distribution in ball
        if uniform_on_radius:
            radii = tf.math.abs(tf.random.uniform((num_points, 1))) * radius
        else:
            radii = tf.math.abs(tf.random.uniform((num_points, 1)))  ** (1/dim) * radius

        # Combine to get points
        points = center + directions * tf.cast(radii, tf.complex64)
        return points

    def poly_normed_abs_val_takes_real(real_points, pg):
        complex_points =point_vec_to_complex(real_points)
        return poly_normed_abs_val(complex_points, pg)

    

    def gradient_descent_vectorized(cpoints, poly_func, pg, learning_rate=0.0001, max_iterations=10000000, max_tolerance=1e-7,kahler_t=1.0):
        """Perform gradient descent to minimize function poly_func for multiple points using vectorized ADAM optimization."""
        realpoints=point_vec_to_real(cpoints)
        print("max_tolerance for gradient descent is "+str(max_tolerance))
        # Initialize the ADAM optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        losses_initial = poly_func(realpoints, pg)
        print(f"Initial mean loss: {tf.reduce_mean(losses_initial):.6e}, max loss: {tf.reduce_max(losses_initial):.6e}")

        # Convert the points to tf.Variable so they're trainable
        realpoints = tf.Variable(realpoints, dtype=tf.float32)

        @tf.function
        def optimize_step():
            with tf.GradientTape() as tape:
                losses = poly_func(realpoints, pg)
            gradients = tape.gradient(losses, realpoints)
            optimizer.apply_gradients([(gradients, realpoints)])
            return losses

        for i in range(1, max_iterations + 1):
            losses = optimize_step()

            if i % 100== 0:
                print(f"Iteration {i}: mean loss = {tf.reduce_mean(losses):.6e}, max loss = {tf.reduce_max(losses):.6e}")
            # Check if max_loss is below certain thresholds and decrease learning rate accordingly
            max_loss = tf.reduce_max(losses)
            if max_loss < 1e-4 and not hasattr(optimizer, 'reduced_once'):
                new_learning_rate = optimizer.learning_rate * 0.1  # Decrease by one order of magnitude
                optimizer.learning_rate.assign(new_learning_rate)
                print(f"Decreased learning rate to {new_learning_rate:.2e}")
                optimizer.reduced_once = True  # Mark that we've reduced the learning rate once
            elif max_loss < 1e-5 and not hasattr(optimizer, 'reduced_twice'):
                new_learning_rate = optimizer.learning_rate * 0.01  # Decrease by two orders of magnitude
                optimizer.learning_rate.assign(new_learning_rate)
                print(f"Decreased learning rate to {new_learning_rate:.2e}")
                optimizer.reduced_twice = True  # Mark that we've reduced the learning rate twice
            # elif max_loss < 1e-7 and not hasattr(optimizer, 'reduced_thrice'):
            #     new_learning_rate = optimizer.learning_rate * 0.01  # Decrease by another two orders of magnitude
            #     optimizer.learning_rate.assign(new_learning_rate)
            #     print(f"Decreased learning rate to {new_learning_rate:.2e}")
            #     optimizer.reduced_thrice = True  # Mark that we've reduced the learning rate thrice

            if max_loss < max_tolerance:
                print(f"Converged at iteration {i}")
                print(f"Final mean loss: {tf.reduce_mean(losses):.6e}, max loss: {max_loss:.6e}")
                break

        return point_vec_to_complex(realpoints.numpy()), losses.numpy()



    initial_points = generate_uniform_ball_c(special_point_complex, radius, num_points_to_generate, uniform_on_radius=uniform_on_radius)
    initial_points= pg._rescale_points(np.array(initial_points))
    # Calculate distances using vectorized geodesic distance for CPn and then the same with final_matrix
    initial_distances_CPn = vectorized_geodesic_distance_CPn(special_point_complex, initial_points, kahler_t=kahler_t)
    initial_distances_matrix = vectorized_geodesic_distance_CPn(special_point_complex, initial_points, kahler_t=kahler_t, metricijbar=final_matrix)
    #delete initial_points if they are too close to special_point_complex
    


    # Perform gradient descent
    optimized_points, final_losses = gradient_descent_vectorized(initial_points, poly_normed_abs_val_takes_real,pg,max_tolerance=max_tolerance_for_gradient_descent)
    optimized_points=tf.constant(pg._rescale_points(optimized_points.numpy()))
    optimized_distances_CPn = vectorized_geodesic_distance_CPn(special_point_complex, optimized_points, kahler_t=kahler_t)
    optimized_distances_matrix = vectorized_geodesic_distance_CPn(special_point_complex, optimized_points, kahler_t=kahler_t, metricijbar=final_matrix)


    mask = optimized_distances_matrix.numpy() > min_radius*kahler_t
    initial_points = tf.boolean_mask(initial_points, mask)
    optimized_points = tf.boolean_mask(optimized_points, mask)

    initial_distances_CPn = tf.boolean_mask(initial_distances_CPn, mask)
    initial_distances_matrix = tf.boolean_mask(initial_distances_matrix, mask)
    optimized_distances_CPn = tf.boolean_mask(optimized_distances_CPn, mask)
    optimized_distances_matrix = tf.boolean_mask(optimized_distances_matrix, mask)


    # Calculate and print the maximum value of poly_normed_abs_val for initial points
    initial_max_val = tf.reduce_max(poly_normed_abs_val(initial_points, pg))
    print(f"Maximum value of poly_normed_abs_val for initial points: {initial_max_val:.6e}")

    # Calculate and print the maximum value of poly_normed_abs_val for optimized points
    optimized_max_val = tf.reduce_max(poly_normed_abs_val(optimized_points, pg))
    print(f"Maximum value of poly_normed_abs_val for optimized points: {optimized_max_val:.6e}")

    # Print statistics for initial and optimized points
    print("CPn distances:")
    print(f"Initial points - Min: {tf.reduce_min(initial_distances_CPn):.10f}, Max: {tf.reduce_max(initial_distances_CPn):.4f}, Mean: {tf.reduce_mean(initial_distances_CPn):.4f}")
    print(f"Optimized points - Min: {tf.reduce_min(optimized_distances_CPn):.10f}, Max: {tf.reduce_max(optimized_distances_CPn):.4f}, Mean: {tf.reduce_mean(optimized_distances_CPn):.4f}")
    print("\nCPn distances with modified metric:")
    print(f"Initial points - Min: {tf.reduce_min(initial_distances_matrix):.10f}, Max: {tf.reduce_max(initial_distances_matrix):.4f}, Mean: {tf.reduce_mean(initial_distances_matrix):.4f}")
    print(f"Optimized points - Min: {tf.reduce_min(optimized_distances_matrix):.10f}, Max: {tf.reduce_max(optimized_distances_matrix):.4f}, Mean: {tf.reduce_mean(optimized_distances_matrix):.4f}")

    points_optimized_to_return=optimized_points[0:num_points]
    #return number of points requested
    print("requested "+str(num_points)+" points, got "+str(len(points_optimized_to_return)))
    return points_optimized_to_return
    



def compute_Gijbar_from_Hijbar(Hijbar,cpoint,kahler_t=1.0):
    H = tf.einsum('iJ,i,J->',Hijbar,cpoint,tf.math.conj(cpoint))
    Hijbar_over_H = Hijbar/H
    HssH_over_H2 = tf.einsum('kJ,k,M,iM->iJ',Hijbar_over_H,cpoint,tf.math.conj(cpoint),Hijbar_over_H)/H**2
    return tf.cast(kahler_t,tf.complex64)*(Hijbar_over_H-HssH_over_H2)


def loss_function(vec, n, cpoint, pullback, g_CY, v_list, weights, kahler_t=1.0):
    """
    Compute the loss function.
    
    Args:
    vec (tf.Tensor): Vector of real numbers.
    n (int): Dimension of the square matrix.
    g (function): Function to apply to the matrix.
    k (tf.Tensor): Target value for g(matrix).
    v_list (list): List of vectors for the orthonormality condition.
    weights (dict): Weights for different components of the loss.
    
    Returns:
    tuple: Total weighted loss, loss1, loss_v, and loss_eig.
    """
    matrix = vector_to_hermitian_matrix(vec, n)
    # Component 1: g(matrix) = k
    Gijbar = compute_Gijbar_from_Hijbar(matrix, cpoint, kahler_t)
    pullback_gijbar = tf.einsum('ai,BJ,iJ->aB', pullback, tf.math.conj(pullback), Gijbar)
    loss1 = tf.reduce_sum(tf.abs(pullback_gijbar - g_CY)**2)
    # Components for orthonormality conditions
    loss_v = tf.reduce_sum([tf.abs(tf.einsum('i,ij,j->', tf.math.conj(v), matrix, v) - (1.0+0j))**2 for v in v_list])
    # Component for positive eigenvalues
    eigenvalues = tf.linalg.eigvals(matrix)
    loss_eig = tf.reduce_sum(tf.nn.relu(-tf.math.real(eigenvalues)))
    # Compute weighted sum of losses
    total_loss = weights['g'] * loss1 + weights['v'] * loss_v + weights['eig'] * loss_eig
    
    return total_loss, loss1, loss_v, loss_eig

def optimize_matrix(rpoint, pullback, g_CY, v_list, weights, learning_rate=0.1, num_epochs=100000, kahler_t=1.0, n_init=50):
    """
    Optimize to find the matrix satisfying the given conditions, using parallel initializations.
    
    Args:
    n (int): Dimension of the square matrix.
    m (int): Length of the input vector.
    rpoint (tf.Tensor): Point in the manifold, real
    pullback (tf.Tensor): Pullback tensor.
    g_CY (tf.Tensor): Target metric.
    v_list (list): List of vectors for the orthonormality condition.
    weights (dict): Weights for different components of the loss.
    learning_rate (float): Learning rate for optimization.
    num_epochs (int): Number of optimization epochs.
    t (float): Parameter for the computation of Gijbar.
    n_init (int): Number of parallel initializations.
    
    Returns:
    tuple: Optimized vector, final loss, satisfaction measure, and lists of losses.
    """
    cpoint = point_vec_to_complex(rpoint)
    n = tf.shape(pullback)[-1]  # Dimension of the square matrix
    m=n**2
    
    # Initialize parameters for all initializations
    initial_params = tf.random.normal((n_init, m))
    vecs = tf.Variable(initial_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    total_losses = []
    losses1 = []
    losses_v = []
    losses_eig = []
    
    @tf.function
    def train_step_batch(vecs):
        with tf.GradientTape() as tape:
            matrix_batch = tf.map_fn(lambda v: vector_to_hermitian_matrix(v, n), vecs, dtype=tf.complex64)
            Gijbar_batch = tf.map_fn(lambda matrix: compute_Gijbar_from_Hijbar(matrix, cpoint, kahler_t), matrix_batch)
            pullback_gijbar_batch = tf.einsum('ai,BJ,niJ->naB', pullback, tf.math.conj(pullback), Gijbar_batch)
            loss1_batch = tf.reduce_sum(tf.abs(pullback_gijbar_batch - g_CY)**2, axis=[1, 2])
            
            # Pre-compute the orthonormality condition for all v in v_list
            loss_v_batch = tf.zeros_like(loss1_batch)
            for v in v_list:
                v_conj = tf.math.conj(v)
                loss_v_batch += tf.abs(tf.einsum('i,nij,j->n', v_conj, matrix_batch, v) - (1.0+0j))**2
            
            eigenvalues_batch = tf.linalg.eigvals(matrix_batch)
            loss_eig_batch = tf.reduce_sum(tf.nn.relu(-tf.math.real(eigenvalues_batch)), axis=1)
            
            total_loss_batch = weights['g'] * loss1_batch + weights['v'] * loss_v_batch + weights['eig'] * loss_eig_batch
        
        gradients = tape.gradient(total_loss_batch, vecs)
        optimizer.apply_gradients([(gradients, vecs)])
        return total_loss_batch, loss1_batch, loss_v_batch, loss_eig_batch#, vecs

    for epoch in range(num_epochs):
        total_loss_batch, loss1_batch, loss_v_batch, loss_eig_batch = train_step_batch(vecs)
        
        total_losses.append(total_loss_batch.numpy())
        losses1.append(loss1_batch.numpy())
        losses_v.append(loss_v_batch.numpy())
        losses_eig.append(loss_eig_batch.numpy())
        
        if epoch % 100 == 0:
            tf.print(f"Epoch {epoch}, Min Total Loss: {tf.reduce_min(total_loss_batch)}")
        
        # Early stopping condition
        if tf.reduce_min(total_loss_batch) < 1e-14:
            tf.print(f"Early stopping at epoch {epoch} with min loss {tf.reduce_min(total_loss_batch)}")
            break

    best_index = tf.argmin(total_loss_batch)
    best_vec = vecs[best_index]
    best_loss = total_loss_batch[best_index]

    final_total_loss, final_loss1, final_loss_v, final_loss_eig = loss_function(best_vec, n, cpoint, pullback, g_CY, v_list, weights, kahler_t)
    satisfaction_measure = 1 / (1 + final_total_loss)  # Maps [0, inf) to (0, 1]
    
    return best_vec, final_total_loss, satisfaction_measure, np.array(total_losses), np.array(losses1), np.array(losses_v), np.array(losses_eig)





def analyze_pullback_kernel(pullback_matrix, point):
    """
    Analyze the kernel of the pullback matrix at a given point.

    This function performs the following steps:
    1. Computes the Singular Value Decomposition (SVD) of the pullback matrix.
    2. Determines the rank of the matrix based on non-zero singular values.
    3. Finds an orthonormal basis for the kernel of the pullback matrix.
    4. Verifies the orthogonality of the kernel basis.
    5. Checks that the kernel basis vectors are indeed in the kernel of the pullback matrix.

    Args:
        pullback_matrix (tf.Tensor): The pullback matrix to analyze.
        point (tf.Tensor): The point at which the pullback matrix is computed.

    Returns:
        tf.Tensor: An orthonormal basis for the kernel of the pullback matrix.

    Prints:
        - The point used for the pullback matrix.
        - The rank of the pullback matrix.
        - The dimension of the kernel.
        - The orthonormal basis for the kernel.
        - Orthogonality check results.
        - Kernel check results.
    """
    # Find an orthogonal basis for the kernel of one of the pullback matrices
    # Compute the SVD of the pullback matrix
    s, u, v = tf.linalg.svd(pullback_matrix, full_matrices=True)

    # Determine the rank of the matrix (number of non-zero singular values)
    # We'll use a small threshold to account for numerical precision
    threshold = 1e-6
    rank = tf.reduce_sum(tf.cast(s > threshold, tf.int32))

    # The last (n - rank) columns of v form an orthonormal basis for the kernel
    # where n is the number of columns in the pullback matrix
    nplus1 = tf.shape(pullback_matrix)[1]
    kernel_basis = v[:, rank:]

    print(f"Point used for pullback matrix:")
    print(point)
    print(f"\nRank of the pullback matrix: {rank}")
    print(f"Dimension of the kernel: {nplus1 - rank}")
    print(f"Orthonormal basis for the kernel:")
    print(tf.transpose(kernel_basis))

    # Verify orthogonality
    orthogonality_check = tf.matmul(tf.math.conj(kernel_basis), kernel_basis, transpose_a=True)
    print("\nOrthogonality check (should be identity matrix):")
    print(orthogonality_check)

    # Verify that these vectors are in the kernel
    kernel_check = tf.matmul(pullback_matrix, kernel_basis)
    print("\nKernel check (should be close to zero):")
    print(tf.reduce_max(tf.abs(kernel_check)))

    return kernel_basis

def optimize_and_get_final_matrix(special_pullback, special_point, metricModel, kahler_t=1.0, plot_losses=False, weights={'g': 1.0, 'v': 1.0, 'eig': 0.1}):
    """
    Optimize the matrix to satisfy given conditions and return the final optimized matrix.

    This function performs the following steps:
    1. Analyzes the pullback kernel to get an orthonormal basis.
    2. Optimizes a vector representation of a Hermitian matrix using the loss function.
    3. Constructs the final Hermitian matrix from the optimized vector.
    4. Optionally plots the loss curves during optimization.

    Args:
        special_pullback (tf.Tensor): The pullback tensor at the special point.
        special_point (tf.Tensor): The coordinates of the special point.
        phimodel1 (callable): A function that computes the CY metric.
        t (float, optional): A scaling factor for the metric. Defaults to 1.0.
        plot_losses (bool, optional): Whether to plot loss curves. Defaults to False.
        weights (dict, optional): Weights for different components of the loss function.
            Defaults to {'g': 1.0, 'v': 1.0, 'eig': 0.1}.

    Returns:
        tf.Tensor: The final optimized Hermitian matrix.

    Raises:
        tf.errors.InvalidArgumentError: If optimization fails due to singular matrices.
    """

    n=tf.shape(special_pullback)[-1]
    kernel_basis = analyze_pullback_kernel(special_pullback, special_point)
    v_list = tf.transpose(kernel_basis)
    g_CY = metricModel(tf.expand_dims(special_point,axis=0))[0]

    while True:
        try:
            optimized_vec, final_loss, satisfaction, total_losses, losses1, losses_v, losses_eig = optimize_matrix(special_point, special_pullback, g_CY, v_list, weights, kahler_t=kahler_t,n_init=50)
            break
        except tf.errors.InvalidArgumentError: #sometimes the optimizer fails, probably due to singular matrices
            print("Optimize matrix failed, trying again")
            continue

    tf.print(f"Optimized vector: {optimized_vec}")
    tf.print(f"Final loss: {final_loss}")
    tf.print(f"Satisfaction measure: {satisfaction}")

    # Construct the final matrix
    final_matrix = vector_to_hermitian_matrix(optimized_vec, n)
    tf.print(f"Final matrix:\n{final_matrix}")

    if plot_losses:
        import matplotlib.pyplot as plt

        # Find the index of the smallest total loss
        best_index = np.argmin(np.array(total_losses[-1]))

        # Print the losses corresponding to the smallest total loss
        print(f"Final losses for the epoch with smallest total loss:")
        print(f"Total Loss: {total_losses[-1,best_index]}")
        print(f"Loss1: {losses1[-1,best_index]}")
        print(f"Loss_v: {losses_v[-1,best_index]}")
        print(f"Loss_eig: {losses_eig[-1,best_index]}")

        # Plot the loss curves
        plt.figure(figsize=(12, 8))
        plt.plot(total_losses[:,best_index], label='Total Loss')
        plt.plot(losses1[:,best_index], label='Loss1')
        plt.plot(losses_v[:,best_index], label='Loss_v')
        plt.plot(losses_eig[:,best_index], label='Loss_eig')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.yscale('log')  # Use log scale for y-axis to better visualize small losses
        plt.grid(True)
        plt.show()

    # Convert final_matrix to numpy, round to 2 significant figures, and print
    numpy_matrix = final_matrix.numpy()
    rounded_matrix = np.round(numpy_matrix, decimals=2)
    print("Rounded final matrix to 2 decimal places:")
    print(rounded_matrix)
    print("eigvals: ",np.round(tf.linalg.eigvals(final_matrix),4))

    actual_metric_locally=compute_Gijbar_from_Hijbar(final_matrix,point_vec_to_complex(special_point),kahler_t=1.0)
    # Pullback the final matrix to the 3x3 matrix
    pulled_back_final_matrix = tf.einsum('ai,BJ,iJ->aB', special_pullback,tf.math.conj(special_pullback), actual_metric_locally)

    # Convert to numpy, round to 2 decimal places, and print
    numpy_pulled_back_matrix = pulled_back_final_matrix.numpy()
    rounded_pulled_back_matrix = np.round(numpy_pulled_back_matrix, decimals=2)
    print("Rounded pulled-back final matrix to 2 decimal places, compared to actual g_CY:")
    print(rounded_pulled_back_matrix)
    print(np.round(g_CY,decimals=2))

    # Compute and print eigenvalues of the pulled-back matrix
    eigenvalues = tf.linalg.eigvals(pulled_back_final_matrix)
    print("Eigenvalues of the pulled-back matrix:")
    print(eigenvalues)

    return final_matrix
