import tensorflow as tf
import numpy as np
import tensorflow.keras as tfk
#def make_nn(n_in,n_out,nlayer,nHidden,act='gelu',lastbias=False,use_zero_network=False):
#   if use_zero_network:
#      nn_phi = tf.keras.Sequential()
#      nn_phi.add(tfk.Input(shape=(n_in,)))
#      for i in range(nlayer):
#          nn_phi.add(tfk.layers.Dense(nHidden, activation=act,kernel_initializer='zeros'))
#      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
#   else:
#      nn_phi = tf.keras.Sequential()
#      nn_phi.add(tfk.Input(shape=(n_in,)))
#      for i in range(nlayer):
#          nn_phi.add(tfk.layers.Dense(nHidden, activation=act))
#      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
#   return nn_phi
#

#testvec=testvec
def symmetrisemat(Xab):
    return Xab + tf.transpose(Xab,perm=[0,2,1])
def antisymmetrisemat(Xab):
    return Xab - tf.transpose(Xab,perm=[0,2,1])
def bihomogeneous_section_for_prod_one_proj(Z):
    X = tf.math.real(Z)
    Y = tf.math.imag(Z)
    
    XX_YY = (tf.einsum('xa,xb->xab',X, X)) + (tf.einsum('xa,xb->xab',Y,Y))
    XY_YX = antisymmetrisemat(tf.einsum('xa,xb->xab',X, Y))
    ones = tf.ones_like(XX_YY[0])
    #zeros= tf.zeros_like(XX_YY[0])
    mask_a = tf.cast(tf.linalg.band_part(ones, 0, -1),dtype=tf.bool) # Upper triangular matrix of 0s and 1s
    mask_b = tf.cast(tf.linalg.band_part(ones, 0, -1)-tf.linalg.band_part(ones, 0, 0),dtype=tf.bool) # Upper triangular matrix of 0s and 1s
    #mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    #mask = tf.cast(mask_a, dtype=tf.bool) # Make a bool mask
    upper_triangular_flat_XX_YY = tf.transpose(tf.boolean_mask(tf.transpose(XX_YY), mask_a,axis=0))
    upper_triangular_flat_XY_YX = -tf.transpose(tf.boolean_mask(tf.transpose(XY_YX), mask_b,axis=0))
    #upper_triangular_flat_XX_YY = tf.where(mask_a,XX_YY, zeros)
    #upper_triangular_flat_XY_YX = tf.where(mask_b,XY_YX, zeros)

    return tf.concat([upper_triangular_flat_XX_YY,upper_triangular_flat_XY_YX],axis=1)#XY_YX#tf.concat([ XX_YY,XY_YX], axis=0)

def _fubini_study_n_potentials( points, t=tf.complex(1., 0.)):
        r"""Computes the Fubini-Study Kahler potential on a single projective
        ambient space factor specified by n.

        Args:
            points (tf.tensor([bSize, ncoords], tf.complex128)): Coordinates of
                the n-th projective spce.
           t (tf.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            tf.tensor([bsize], tf.float64):
                FS-metric in the ambient space coordinates.
        """
        point_square = tf.math.reduce_sum(tf.math.abs(points)**2, axis=-1)
        return tf.cast(t/np.pi, tf.float64) * tf.cast(tf.math.log(point_square), tf.float64)

def getrealandimagofprod(cpoints,return_mat=False):
    X = tf.math.real(cpoints)
    Y = tf.math.imag(cpoints)
    XX_YY = (tf.einsum('xa,xb->xab',X, X)) + (tf.einsum('xa,xb->xab',Y,Y))
    XY_YX = antisymmetrisemat(tf.einsum('xa,xb->xab',X, Y))
    ones = tf.ones_like(XX_YY[0])
    #zeros= tf.zeros_like(XX_YY[0])
    mask_a = tf.cast(tf.linalg.band_part(ones, 0, -1),dtype=tf.bool) # Upper triangular matrix of 0s and 1s
    mask_b = tf.cast(tf.linalg.band_part(ones, 0, -1)-tf.linalg.band_part(ones, 0, 0),dtype=tf.bool) # Upper triangular matrix of 0s and 1s
    #mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    #mask = tf.cast(mask_a, dtype=tf.bool) # Make a bool mask
    if return_mat:
        real = XX_YY
        imag = XY_YX
    else:
        real = tf.transpose(tf.boolean_mask(tf.transpose(XX_YY), mask_a,axis=0))
        imag = -tf.transpose(tf.boolean_mask(tf.transpose(XY_YX), mask_b,axis=0))#minus sign to make it XYbar - Y Xbar, or the other way around?    
    return real, imag


class bihom_function_generator_old(tf.Module):
    def __init__(self, AmbientEnvVarToBeSet,nProjectiveToBeSet,kmoduli):
        super().__init__()
        self.AmbientEnvVarToBeSet = AmbientEnvVarToBeSet
        self.nProjectiveToBeSet= nProjectiveToBeSet
        self.kmoduli= kmoduli

    @tf.function
    def __call__(self,points):
        r"""Computes the Kahler potential.

        Args:
            points ( # NO - TAKES COMPLEX INPUT


        """
        #ambient=np.abs(BASIS['AMBIENT'].numpy()).astype(int)

        #ambient=tf.cast((ambientBASIS,tf.int32))
        ambient=self.AmbientEnvVarToBeSet
        #print('test')
        #nProjective=(np.shape(ambient))[0]#.astype(int)#nprojective spaces
        #nProjective=(len(ambient))#.astype(int)#nprojective spaces
        nProjective = self.nProjectiveToBeSet#tf.shape(ambient)[0]
        # print("NPROJECTIVE")
        # print(nProjective)
        #print('nProjective '+str(nProjective))
        #if nProjective > 1:
        # we go through each ambient space factor and create the Kahler potential.
        degrees=ambient+1
        # print('deg')
        # print(degrees)
        #print('degsss')
        #print(degrees)
        #print(degrees.numpy())
        #ncoords=tf.reduce_sum(ambient+1)
        #degrees[0]
        cpoints = points[:, :degrees[0]]
        #print("here")
        kappas = tf.math.reduce_sum(tf.math.abs(cpoints)**2, axis=-1)

        #print("here")
        #k_fs = _fubini_study_n_potentials(cpoints, t=BASIS['KMODULI'][0])
        k_fs = _fubini_study_n_potentials(cpoints, t=self.kmoduli[0])
        #print("here")
        #cpoints_stored=[cpoints]
        #cpoints_stored=tf.TensorArray(tf.complex128,size=nProjective)
        #cpoints_stored=cpoints_stored.write(0,cpoints)
        kappasprod=kappas
        #print('degs '+ str(degrees))
        #print('nProjec'+ str(nProjective))
        for i in range(1, nProjective):
            indices = tf.range(i)
            s = tf.reduce_sum(tf.gather(degrees, indices))
            #s = tf.reduce_sum(degrees[0:i])
            e = s + degrees[i]
            cpoints = points[:, s:e]
            #print("here1")
            #k_fs_tmp = _fubini_study_n_potentials(cpoints, t=BASIS['KMODULI'][i])
            k_fs_tmp = _fubini_study_n_potentials(cpoints, t=self.kmoduli[i])
            #print("here1")
            kappas = tf.math.reduce_sum(tf.math.abs(cpoints)**2, axis=-1)
            #print("here1")
            k_fs += k_fs_tmp
            kappasprod *= kappas
            #print(tf.shape(cpoints))
            #print(len(cpoints_stored))

            #cpoints_stored+=[cpoints]
            #cpoints_stored=cpoints_stored.write(i,cpoints)
        #print("here2")
        #zzbar=[]
        indices = tf.range(0)
        s0 = tf.reduce_sum(tf.gather(degrees, indices))
        e0= s0 + degrees[0]
        #cpoints0 = 
        iterativereal,iterativeimag=getrealandimagofprod(points[:, s0:e0],return_mat=False)
        iterativerealsize=int(degrees[0]*(degrees[0]+1)/2)
        iterativeimagsize=int(degrees[0]*(degrees[0]-1)/2)
        # tf.autograph.experimental.set_loop_options(
        #             shape_invariants=[(iterativeimag, tf.TensorShape([None])),(iterativereal, tf.TensorShape([None]))]
        #         )
        #print("here3")
        for i in range(1,nProjective):
            indices = tf.range(i)
            s = tf.reduce_sum(tf.gather(degrees, indices))
            #s = tf.reduce_sum(degrees[0:i])
            e = s + degrees[i]
            cpoints = points[:, s:e]
            #zzbar+=tf.einsum('xi,xj->xij',cpoints_stored[i],tf.math.conj(cpoints_stored[i]))
            #print("here4")
            realtoadd,imagtoadd = getrealandimagofprod(cpoints,return_mat=False)   
            tempimagsize=int(degrees[i]*(degrees[i]-1)/2)
            temprealsize=int(degrees[i]*(degrees[i]+1)/2)
            iterativerealparts=[iterativerealsize*temprealsize,iterativeimagsize*tempimagsize]
            iterativeimagparts=[iterativerealsize*tempimagsize,iterativeimagsize*temprealsize]
            iterativereal2 = tf.concat([tf.reshape(tf.einsum('xa,xi->xai', iterativereal, realtoadd),[-1,iterativerealparts[0]]), 
                                             - tf.reshape(tf.einsum('xa,xi->xai', iterativeimag, imagtoadd),[-1,iterativerealparts[1]])],axis=-1)#tf.einsum('xa,xi->aijkl', iterativeimag, imagtoadd),[-1,]
            iterativeimag2 = tf.concat([tf.reshape(tf.einsum('xa,xi->xai', iterativereal, imagtoadd),[-1,iterativeimagparts[0]]), 
                                             - tf.reshape(tf.einsum('xa,xi->xai', iterativeimag, realtoadd),[-1,iterativeimagparts[1]])],axis=-1)#tf.einsum('xa,xi->aijkl', iterativeimag, imagtoadd),[-1,]
            iterativerealsize=iterativerealparts[0]+iterativerealparts[1]
            iterativeimagsize=iterativeimagparts[0]+iterativeimagparts[1]
            iterativereal = iterativereal2
            iterativeimag = iterativeimag2
            #iterativeimag2 = tf.einsum('aij,akl->aijkl', iterativereal, imagtoadd) + tf.einsum('aij,akl->aijkl', iterativeimag, realtoadd)
        sectimessecbar_over_kappa= tf.einsum('xi,x->xi',tf.concat([iterativereal,iterativeimag],axis=-1),1/kappasprod)
        return sectimessecbar_over_kappa


class bihom_function_generatorQorT(tf.Module):
    def __init__(self, ambient_env_var, n_projective, kmoduli):
        super().__init__()
        self.ambient_env_var = ambient_env_var
        self.n_projective = n_projective
        self.kmoduli = kmoduli
        self.degrees = self.ambient_env_var + 1
        self.cumsum_degrees = tf.cumsum(self.degrees)
        
        # Corrected slice_indices calculation
        starts = tf.concat([[0], self.cumsum_degrees[:-1]], axis=0)
        ends = self.cumsum_degrees
        self.slice_indices = tf.stack([starts, ends], axis=1)
    
    @tf.function
    def __call__(self, points):
        #takes complex points
        #k_fs = tf.zeros_like(points[:, 0], dtype=tf.float32)
        #sqrtkappas = tf.math.sqrt(tf.reduce_sum(tf.abs(points[0:2])**2, axis=-1)*tf.reduce_sum(tf.abs(points[2:4])**2, axis=-1)*tf.reduce_sum(tf.abs(points[4:6])**2, axis=-1)*tf.reduce_sum(tf.abs(points[6:8])**2, axis=-1))
        points0=tf.einsum('xi,x->xi',points,tf.cast(tf.reduce_sum(tf.abs(points)**2, axis=-1)**(-0.5),tf.complex128))
        secsecbar_r,secsecbar_i=getrealandimagofprod(points0)
        secsecbar=tf.concat((secsecbar_r,secsecbar_i),axis=1)
        return secsecbar

class bihom_function_generator(tf.Module):
    def __init__(self, ambient_env_var, n_projective, kmoduli):
        super().__init__()
        self.ambient_env_var = ambient_env_var
        self.n_projective = n_projective
        self.kmoduli = kmoduli
        self.degrees = self.ambient_env_var + 1
        self.cumsum_degrees = tf.cumsum(self.degrees)
        
        # Corrected slice_indices calculation
        starts = tf.concat([[0], self.cumsum_degrees[:-1]], axis=0)
        ends = self.cumsum_degrees
        self.slice_indices = tf.stack([starts, ends], axis=1)
    
          # Precompute shapes using an example vector
        self._precompute_shapes()

    def _precompute_shapes(self):
        # Create an example vector
        points = tf.complex(
            tf.ones((1, tf.reduce_sum(self.degrees))),
            tf.zeros((1, tf.reduce_sum(self.degrees)))
        )

        self.shapes = []

        for i in range(self.n_projective):
            s = self.slice_indices[i,0]
            e = self.slice_indices[i,1]
            cpoints = points[:, s:e]
            
            real, imag = getrealandimagofprod(cpoints)
            
            if i == 0:
                iterative_real, iterative_imag = real, imag
            else:
                iterative_real, iterative_imag = self._update_iterative_for_precomputing(iterative_real, iterative_imag, real, imag)
            
            #self.shapes.append(tf.constant({
            #    'real': tf.shape(real),
            #    'imag': tf.shape(imag),
            #    'iterative_real': tf.shape(iterative_real),
            #    'iterative_imag': tf.shape(iterative_imag)
            #}))
            self.shapes.append([
                tf.shape(real)[1],
                tf.shape(imag)[1],
                tf.shape(iterative_real)[1],
                tf.shape(iterative_imag)[1]
            ])



        # Compute final output shape
        #final_shape = tf.shape(tf.concat([iterative_real, iterative_imag], axis=-1))
        #self.shapes.append({'final': final_shape})
        self.shapes=tf.constant(np.array(self.shapes))

    @tf.function
    def __call__(self, points):
        #k_fs = tf.zeros_like(points[:, 0], dtype=tf.float64)
        k_fs=tf.zeros([tf.shape(points)[0]],dtype=tf.float64)
        #kappas_prod = tf.ones_like(points[:, 0], dtype=tf.float64)
        kappas_prod=tf.ones([tf.shape(points)[0]],dtype=tf.float64)
        iterative_real = tf.zeros_like(points[:, 0:1], dtype=tf.float64)
        iterative_imag = tf.zeros_like(points[:, 0:1], dtype=tf.float64)

        for i in range(self.n_projective):
            #print("shaep")
            #print(self.shapes[i])
            s = self.slice_indices[i,0]
            e = self.slice_indices[i,1]
            cpoints = points[:, s:e]
            
            k_fs += _fubini_study_n_potentials(cpoints, self.kmoduli[i])
            kappas = tf.reduce_sum(tf.abs(cpoints)**2, axis=-1)
            #tf.print("dtypes of kappas, cpoints, points, etc.")
            #tf.print(kappas.dtype, cpoints.dtype, points.dtype)
            #print("dtypes of kappas, cpoints, points, etc.")
            #print(kappas.dtype, cpoints.dtype, points.dtype)
        
            kappas_prod *= kappas

            real, imag = getrealandimagofprod(cpoints)
            
            if i == 0:
                iterative_real, iterative_imag = real, imag
            else:
                iterative_real, iterative_imag = self._update_iterative_saved(iterative_real, iterative_imag, real, imag, self.shapes[i-1])

        sec_times_secbar_over_kappa = tf.concat([iterative_real, iterative_imag], axis=-1) / kappas_prod[:, tf.newaxis]
        return sec_times_secbar_over_kappa

        

    @tf.function
    def _update_iterative_for_precomputing(self, iterative_real, iterative_imag, real, imag):
        real_size = tf.shape(real)[1]
        imag_size = tf.shape(imag)[1]
        
        iterative_real_size = tf.shape(iterative_real)[1]
        iterative_imag_size = tf.shape(iterative_imag)[1]
        
        real_parts = [iterative_real_size * real_size, iterative_imag_size * imag_size]
        imag_parts = [iterative_real_size * imag_size, iterative_imag_size * real_size]
        
        new_real = tf.concat([
            tf.reshape(tf.einsum('bi,bj->bij', iterative_real, real), [-1, real_parts[0]]),
            -tf.reshape(tf.einsum('bi,bj->bij', iterative_imag, imag), [-1, real_parts[1]])
        ], axis=1)
        
        new_imag = tf.concat([
            tf.reshape(tf.einsum('bi,bj->bij', iterative_real, imag), [-1, imag_parts[0]]),
            -tf.reshape(tf.einsum('bi,bj->bij', iterative_imag, real), [-1, imag_parts[1]])
        ], axis=1)
        
        return new_real, new_imag

    @tf.function
    def _update_iterative_saved(self, iterative_real, iterative_imag, real, imag,shapes):
        # real_size = tf.shape(real)[1]
        # imag_size = tf.shape(imag)[1]
        
        # iterative_real_size = tf.shape(iterative_real)[1]
        # iterative_imag_size = tf.shape(iterative_imag)[1]
        real_size=shapes[0]
        imag_size=shapes[1]
        iterative_real_size=shapes[2]
        iterative_imag_size=shapes[3]
        
        real_parts = [iterative_real_size * real_size, iterative_imag_size * imag_size]
        imag_parts = [iterative_real_size * imag_size, iterative_imag_size * real_size]
        
        new_real = tf.concat([
            tf.reshape(tf.einsum('bi,bj->bij', iterative_real, real), [-1, real_parts[0]]),
            -tf.reshape(tf.einsum('bi,bj->bij', iterative_imag, imag), [-1, real_parts[1]])
        ], axis=1)
        
        new_imag = tf.concat([
            tf.reshape(tf.einsum('bi,bj->bij', iterative_real, imag), [-1, imag_parts[0]]),
            -tf.reshape(tf.einsum('bi,bj->bij', iterative_imag, real), [-1, imag_parts[1]])
        ], axis=1)
        
        return new_real, new_imag
    
def get_monomial_indices(example,degree):
    if degree==0:
        return np.array([])
    """
    Computes the vector of n-degree monomials of the variables in the given vector.

    Args:
        variables: A 1D tensor of arbitrary length representing the variables.
        degree: An integer representing the degree of the monomials.

    Returns:
        A 1D tensor containing the n-degree monomials of the variables.# really 1D?
    """
    # Get the number of variables
    num_vars = tf.shape(example)[-1]
    indices_arr=tf.stack(
                tf.meshgrid(
                    *[tf.range(num_vars) for _ in range(degree)],
                    indexing='ij'
                ),axis=-1
            )

            
    # Generate all possible combinations of variables with repetition
    # indices = tf.transpose(
    #     tf.reshape(
    #         indices_arr,
    #         (degree, -1)
    #     )
    # )


    #print(indices_arr)    
    #print("indices_arar")
    unique_indices=flatten_upper_triangular(indices_arr)
    reshaped_indices = tf.reshape(unique_indices, (-1, degree))
    return reshaped_indices

def monomials(degree,cpoints):
    # Gather the variables for each unique combination of indices
    if degree==0:
        return tf.ones_like(cpoints[:,0:1])#returna  list of a single variable, which is 1
    reshaped_indices=get_monomial_indices(cpoints[0],degree)
    gathered_pts= tf.gather(cpoints, reshaped_indices,axis=-1)
    # Compute the monomials by multiplying the variables along the last axis
    monomials_tensor = tf.reduce_prod(gathered_pts, axis=-1)

    return monomials_tensor

# def empty_gather_on_last_axis(points, indices):
#     # Get the shape of the params tensor

#     # Get the number of dimensions in the params tensor

#     # Create an empty tensor with the same dtype as params
#     #empty_tensor = tf.expand_dims(tf.zeros([params_shape[i] for i in range(num_dims - 1)] + [0], dtype=params.dtype),axis=-1)
#     empty_tensor = tf.expand_dims(tf.zeros_like(points, dtype=points.dtype)[:,0:0],axis=-1)

#     # Use tf.cond to handle empty indices
#     result = tf.cond(tf.equal(tf.size(indices), 0),
#                      lambda: empty_tensor,
#                      lambda: tf.gather(points, indices, axis=-1))

#     return result


def empty_gather_on_last_axis(points, indices):
    # This implementation will safely handle empty indices and avoid pitfalls with tf.cond in gradient computations.

    # Create a boolean to check if indices are empty
    indices_empty = tf.equal(tf.size(indices), 0)

    # Handling for the empty index case
    if indices_empty:
        # Generate an empty tensor by slicing `points` tensor to have zero elements along the needed dimension
        points_shape = tf.shape(points)
        empty_tensor = tf.zeros(shape=tf.concat([points_shape[:-1], [0]], axis=0), dtype=points.dtype)
        empty_tensor = tf.expand_dims(empty_tensor, axis=-1)
        return empty_tensor

    # Non-empty case - directly gather on the last axis
    else:
        return tf.gather(points, indices, axis=-1)


# def monomialsWithMeta(cpoints,indices_to_take_reshaped):
#     # Gather the variables for each unique combination of indices
#     #if degree==0:
#     #    return tf.ones_like(cpoints[:,0:1])#returna  list of a single variable, which is 1
#     #reshaped_indices=get_monomial_indices(cpoints[0],degree)
#     reshaped_indices= indices_to_take_reshaped
#     #print("resh")
#     #print(reshaped_indices)
#     gathered_pts= empty_gather_on_last_axis(cpoints, reshaped_indices)#,axis=-1)
#     #print(gathered_pts.shape)
#     matrixNby1ofOnes=tf.cast(tf.ones_like(cpoints[:,0:1]),tf.complex128)
#     onescomplex=tf.expand_dims(tf.repeat(matrixNby1ofOnes,tf.shape(gathered_pts)[-1],axis=1),axis=1)
#     #onescomplex=tf.expand_dims(tf.repeat(matrixNby1ofOnes,tf.shape(reshaped_indices)[0],axis=1),axis=-1)
#     #print(onescomplex.shape)
#     #add_ones=tf.concat((gathered_pts,onescomplex),axis=1)

#     #print(add_ones.shape)
#     #shape to limit to
#     #getridofonesifnecessary=tf.maximum(tf.shape(gathered_pts)[1], 1)
#     # Compute the monomials by multiplying the variables along the last axis
#     #monomials_tensor = tf.reduce_prod(add_ones, axis=-1)[:,:getridofonesifnecessary]
#     tf.print(tf.shape(cpoints))
#     tf.print(tf.shape(onescomplex))
#     tf.print(tf.shape(gathered_pts))
#     tf.print(len(indices_to_take_reshaped))
#     if len(indices_to_take_reshaped)!=0:
#         monomials_tensor= tf.reduce_prod(gathered_pts, axis=-1)
#     else:
#         monomials_tensor = tf.reduce_prod(onescomplex, axis=-1)
#     #print(monomials_tensor.shape)
#     #print(tf.ones_like(cpoints[:,0:1]))
#     tf.print(tf.shape(monomials_tensor))

#     return monomials_tensor

def monomialsWithMeta(cpoints,indices_to_take_reshaped):
    # Gather the variables for each unique combination of indices
    #if degree==0:
    if len(indices_to_take_reshaped)==0:
        return tf.ones_like(cpoints[:,0:1])#returna  list of a single variable, which is 1
    #reshaped_indices=get_monomial_indices(cpoints[0],degree)
    reshaped_indices= indices_to_take_reshaped
    gathered_pts= tf.gather(cpoints, reshaped_indices,axis=-1)
    # Compute the monomials by multiplying the variables along the last axis
    monomials_tensor = tf.reduce_prod(gathered_pts, axis=-1)

    return monomials_tensor

from functools import reduce
import operator
def product_of_tensor_lengths(tensor_list):
    return reduce(operator.mul, (tf.shape(tensor)[0] for tensor in tensor_list), 1)

class get_degree_kphiandMmonomials_general(tf.Module):
    def __init__(self,kphi,linebundleindices,ambient_env_var,n_projective,kmoduli):
        #define inds
        self.ambient_env_var = ambient_env_var
        self.degrees = self.ambient_env_var + 1
        indskpModM=[tf.cast(get_monomial_indices(tf.ones(self.degrees[i]),kphi[i]+tf.math.abs(linebundleindices[i])),tf.int32) for i in range((kphi.shape)[-1])]
        indsk=[tf.cast(get_monomial_indices(tf.ones(self.degrees[i]),kphi[i]),tf.int32) for i in range(len(kphi))] #conj is unnecessary here
        #print(indsk)
        self.indslist=(indskpModM,indsk)
        self.kphi=kphi
        self.linebundleindices=linebundleindices
        
        self.n_projective = n_projective
        self.kmoduli = kmoduli
        self.cumsum_degrees = tf.cumsum(self.degrees)
        
        # Corrected slice_indices calculation
        starts = tf.concat([[0], self.cumsum_degrees[:-1]], axis=0)
        ends = self.cumsum_degrees
        self.slice_indices = tf.stack([starts, ends], axis=1)
        #tf.print(indsk)
        #tf.print(indskpModM)
        self.n_holo_secs=product_of_tensor_lengths(indskpModM)
        self.n_antiholo_secs=product_of_tensor_lengths(indsk)
        self.n_secs = self.n_holo_secs*self.n_antiholo_secs

    @tf.function
    def __call__(self,cpoints):
        indskpModM_all,indsk_all=self.indslist
        #what is the shape of inds?
        # result=tf.Array()
        for i in range(0,len(self.linebundleindices)):
            indskpModM=indskpModM_all[i]
            indsk=indsk_all[i]

            #indskpModM=get_monomial_indices(cpoints[0,0:2],kphi[i]+tf.math.abs(linebundleindices[i]))
            #indsk=get_monomial_indices(cpoints[0,0:2],kphi[i])#conj is unnecessary here
            s = self.slice_indices[i,0]
            e= self.slice_indices[i,1]
            kappa_i=tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,s:e])**2,axis=-1),tf.complex128)
            if self.linebundleindices[i]>=0:
                #kappa_i_kphi=kappa_i**self.kphi[i]
                #monsbar=monomialsWithMeta(kphi[i],tf.math.conj(cpoints[:,2*i:2*i+2]),indsk)
                #mons=monomialsWithMeta(kphi[i]+np.abs(linebundleindices[i]),cpoints[:,2*i:2*i+2],indskpModM)
                mons=monomialsWithMeta(cpoints[:,s:e],indskpModM)
                monsbar=monomialsWithMeta(tf.math.conj(cpoints[:,s:e]),indsk)

                kappa_i_kphi=kappa_i**self.kphi[i]
                #print(mons.dtype)
                #print(monsbar.dtype)
                #rint(kappa_i_kphi.dtype)
                outer_prod_of_mons_and_monsbar=tf.einsum('xi,xj,x->xij',mons,monsbar,1/kappa_i_kphi)
                #tf.print('test',self.linebundleindices[i],indsk)
                #tf.print('test',self.linebundleindices[i],indskpModM)
                #tf.print(tf.shape(outer_prod_of_mons_and_monsbar))
                outerresult=tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]*tf.shape(outer_prod_of_mons_and_monsbar)[-2]))
            elif self.linebundleindices[i]<0:
                #mons=monomialsWithMeta(kphi[i],cpoints[:,2*i:2*i+2],indsk)
                #monsbar=monomialsWithMeta(kphi[i]+np.abs(linebundleindices[i]),tf.math.conj(cpoints[:,2*i:2*i+2]),indskpModM)
                mons=monomialsWithMeta(cpoints[:,s:e],indsk)
                monsbar=monomialsWithMeta(tf.math.conj(cpoints[:,s:e]),indskpModM)
                #print(monsbar)
                #print(mons)
                kappa_i_kphiplusmodM=kappa_i**tf.cast((self.kphi[i]+tf.math.abs(self.linebundleindices[i])),tf.complex128)
                outer_prod_of_mons_and_monsbar=tf.einsum('xi,xj,x->xij',mons,monsbar,1/kappa_i_kphiplusmodM)
                #tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]))
                #tf.print('test')
                #tf.print('test',self.linebundleindices[i],indsk)
                #tf.print('test',self.linebundleindices[i],indskpModM)
                #print(tf.size(outer_prod_of_mons_and_monsbar))
                #tf.print(tf.shape(outer_prod_of_mons_and_monsbar))
                outerresult=tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]*tf.shape(outer_prod_of_mons_and_monsbar)[-2]))
            #take outer product of outerresult and the previous result, for each i
            if i==0:
                result=outerresult
                #result=result.write(0,outerresult)
            else:
                result=tf.einsum('xi,xj->xij',result,outerresult)
                result=tf.reshape(result,(-1,tf.shape(result)[-1]*tf.shape(result)[-2]))
        return result

def get_degree_kphiandMmonomials_func(kphi,linebundleindices,indslist,cpoints):
        indskpModM_all,indsk_all=indslist
        #what is the shape of inds?
        # result=tf.Array()
        for i in range(0,len(linebundleindices)):
            indskpModM=indskpModM_all[i]
            indsk=indsk_all[i]

            #indskpModM=get_monomial_indices(cpoints[0,0:2],kphi[i]+tf.math.abs(linebundleindices[i]))
            #indsk=get_monomial_indices(cpoints[0,0:2],kphi[i])#conj is unnecessary here

            kappa_i=tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,2*i:2*i+2])**2,axis=-1),tf.complex128)
            if linebundleindices[i]>=0:
                #kappa_i_kphi=kappa_i**kphi[i]
                #monsbar=monomialsWithMeta(kphi[i],tf.math.conj(cpoints[:,2*i:2*i+2]),indsk)
                #mons=monomialsWithMeta(kphi[i]+np.abs(linebundleindices[i]),cpoints[:,2*i:2*i+2],indskpModM)
                mons=monomialsWithMeta(cpoints[:,2*i:2*i+2],indskpModM)
                monsbar=monomialsWithMeta(tf.math.conj(cpoints[:,2*i:2*i+2]),indsk)

                kappa_i_kphi=kappa_i**kphi[i]
                outer_prod_of_mons_and_monsbar=tf.einsum('xi,xj,x->xij',mons,monsbar,1/kappa_i_kphi)
                #tf.print('test',linebundleindices[i],indsk)
                #tf.print('test',linebundleindices[i],indskpModM)
                #tf.print(tf.shape(outer_prod_of_mons_and_monsbar))
                outerresult=tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]*tf.shape(outer_prod_of_mons_and_monsbar)[-2]))
            elif linebundleindices[i]<0:
                #mons=monomialsWithMeta(kphi[i],cpoints[:,2*i:2*i+2],indsk)
                #monsbar=monomialsWithMeta(kphi[i]+np.abs(linebundleindices[i]),tf.math.conj(cpoints[:,2*i:2*i+2]),indskpModM)
                mons=monomialsWithMeta(cpoints[:,2*i:2*i+2],indsk)
                monsbar=monomialsWithMeta(tf.math.conj(cpoints[:,2*i:2*i+2]),indskpModM)
                #print(monsbar)
                #print(mons)
                kappa_i_kphiplusmodM=kappa_i**tf.cast((kphi[i]+tf.math.abs(linebundleindices[i])),tf.complex128)
                outer_prod_of_mons_and_monsbar=tf.einsum('xi,xj,x->xij',mons,monsbar,1/kappa_i_kphiplusmodM)
                #tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]))
                #tf.print('test')
                #tf.print('test',linebundleindices[i],indsk)
                #tf.print('test',linebundleindices[i],indskpModM)
                #print(tf.size(outer_prod_of_mons_and_monsbar))
                #tf.print(tf.shape(outer_prod_of_mons_and_monsbar))
                outerresult=tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]*tf.shape(outer_prod_of_mons_and_monsbar)[-2]))
            #take outer product of outerresult and the previous result, for each i
            if i==0:
                result=outerresult
                #result=result.write(0,outerresult)
            else:
                result=tf.einsum('xi,xj->xij',result,outerresult)
                result=tf.reshape(result,(-1,tf.shape(result)[-1]*tf.shape(result)[-2]))
        return result

# def get_degree_kphiandMmonomials(kphi,linebundleindices,cpoints,indslist):
#     indskpModM_all,indsk_all=indslist
#     result=tf.Array()
#     for i in range(0,len(linebundleindices)):
#         indskpModM=indskpModM_all[i]
#         indsk=indsk_all[i]

#         #indskpModM=get_monomial_indices(cpoints[0,0:2],kphi[i]+tf.math.abs(linebundleindices[i]))
#         #indsk=get_monomial_indices(cpoints[0,0:2],kphi[i])#conj is unnecessary here

#         kappa_i=tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,2*i:2*i+2])**2,axis=-1),tf.complex128)
#         if linebundleindices[i]>=0:
#             kappa_i_kphi=kappa_i**kphi[i]
#             #monsbar=monomialsWithMeta(kphi[i],tf.math.conj(cpoints[:,2*i:2*i+2]),indsk)
#             #mons=monomialsWithMeta(kphi[i]+np.abs(linebundleindices[i]),cpoints[:,2*i:2*i+2],indskpModM)
#             mons=monomialsWithMeta(cpoints[:,2*i:2*i+2],indskpModM)
#             monsbar=monomialsWithMeta(tf.math.conj(cpoints[:,2*i:2*i+2]),indsk)

#             kappa_i_kphi=kappa_i**kphi[i]
#             outer_prod_of_mons_and_monsbar=tf.einsum('xi,xj,x->xij',mons,monsbar,1/kappa_i_kphi)
#             outerresult=tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]*tf.shape(outer_prod_of_mons_and_monsbar)[-2]))
#         elif linebundleindices[i]<0:
#             #mons=monomialsWithMeta(kphi[i],cpoints[:,2*i:2*i+2],indsk)
#             #monsbar=monomialsWithMeta(kphi[i]+np.abs(linebundleindices[i]),tf.math.conj(cpoints[:,2*i:2*i+2]),indskpModM)
#             mons=monomialsWithMeta(cpoints[:,2*i:2*i+2],indsk)
#             monsbar=monomialsWithMeta(tf.math.conj(cpoints[:,2*i:2*i+2]),indskpModM)
#             #print(monsbar)
#             #print(mons)
#             kappa_i_kphiplusmodM=kappa_i**tf.cast((kphi[i]+tf.math.abs(linebundleindices[i])),tf.complex128)
#             outer_prod_of_mons_and_monsbar=tf.einsum('xi,xj,x->xij',mons,monsbar,1/kappa_i_kphiplusmodM)
#             #tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]))
#             outerresult=tf.reshape(outer_prod_of_mons_and_monsbar,(-1,tf.shape(outer_prod_of_mons_and_monsbar)[-1]*tf.shape(outer_prod_of_mons_and_monsbar)[-2]))
#         #take outer product of outerresult and the previous result, for each i
#         if i==0:
#             #result=outerresult
#             result=result.write(0,outerresult)
#         else:
#             result=result.write(0,tf.einsum('xi,xj->xij',result.read(0),outerresult))
#             result= result.write(0,tf.reshape(result.read(0),(-1,tf.shape(result.read(0))[-1]*tf.shape(result.read(0))[-2])))
#     return result



# def flatten_upper_triangular(arr):
#     # Get the shape of the input array
#     shape = tf.shape(arr)
    
#     # Get the number of dimensions
#     ndims = tf.shape(shape)[0]
    
#     # Create a mask for the upper triangular part
#     mask = tf.greater_equal(tf.range(shape[0]), tf.range(shape[0])[:, tf.newaxis])
#     print('mask')
#     print(mask)
    
#     # Recursively apply the mask to higher dimensions
#     for i in range(1, ndims):
#         mask = tf.logical_and(mask, tf.greater_equal(tf.range(shape[i]), tf.range(shape[i])[:, tf.newaxis]))
#     print(mask) 
#     # Apply the mask to the input array
#     upper_triangular = tf.boolean_mask(arr, mask)
    
#     return upper_triangular


def flatten_upper_triangular(arr):
    # Get the shape of the input array
    shape = tf.shape(arr)[:-1]
    
    # Get the number of dimensions
    ndims = tf.shape(shape)[0]
    if ndims==1:
        return arr
    
    # Create a list of indices for each dimension
    indices = tf.meshgrid(*[tf.range(s) for s in shape], indexing='ij')
    #print('indi0ces')
    #print(indices)
    # Create a mask for the upper triangular part
    mask = tf.greater_equal(indices[0], indices[1])
    
    # Recursively apply the mask to higher dimensions
    for i in range(2, ndims):
        mask = tf.logical_and(mask, tf.greater_equal(indices[i-1], indices[i]))
    
    #rint('mask')
    #print(mask)
    
    # Apply the mask to the input array
    upper_triangular = tf.boolean_mask(arr, mask)
    
    return upper_triangular



if __name__ == "__main__":
    # Define the variables
    #variables = tf.constant([1+0.j,])

    # Specify the degree of the monomials
    # degree = 

    # # Compute the monomials
    # monomials_tensor = monomials(variables, degree)

    # print("Monomials:", monomials_tensor)
    kphi=np.array([0,0,0,0])
    linebundleindices=np.array([0,2,-2,0])
    cpoints1=tf.constant([[1.+1.j,2. +2j,3.+3j,1.,1.,2-1.j,7.,8.]]*3)
    cpoints2=tf.constant([[1.+1.j,2.+2j,3.+3j,1.,1./(2-1.j),1.,7.,8.]]*3)
    #need a sufficient basis if
    #print(get_degree_kphiandMmonomials(kphi,np.array([0,1,-1,0]),tf.constant([[1.+1.j,2.+2j,3.+3j,4.,5.,6.,7.,8.]]*3)))
    indskpModM=[get_monomial_indices(tf.ones((2)),kphi[i]+tf.math.abs(linebundleindices[i])) for i in range(len(kphi))]
    indsk=[get_monomial_indices(tf.ones(2),kphi[i]) for i in range(len(kphi))] #conj is unnecessary here
    print(indsk)
    indslist=(indskpModM,indsk)
    # print(get_degree_kphiandMmonomials(kphi,linebundleindices,cpoints1,indslist))
    # print((2-1.j)**(-2)*get_degree_kphiandMmonomials(kphi,linebundleindices,cpoints2,indslist))



def bihomogeneous_section_for_prod_not_mult(points,BASIS):
    r"""Computes the Kahler potential.

    Args:
        points (tf.tensor([bSize, 2*ncoords], tf.float64)): Points.

    Returns:
        tf.tensor([bSize], tf.float64): Kahler potential.

    
    """
    ambient=np.abs(BASIS['AMBIENT'].numpy()).astype(int)
    #print('test')
    nProjective=(np.shape(ambient))[0]#.astype(int)#nprojective spaces
    #print('nProjective '+str(nProjective))
    #if nProjective > 1:
    # we go through each ambient space factor and create the Kahler potential.
    degrees=ambient+1
    #print('degsss')
    #print(degrees)
    #print(degrees.numpy())
    ncoords=tf.reduce_sum(ambient+1)
    #degrees[0]
    cpoints = points[:, :degrees[0]]
    #print("here")
    kappas = tf.math.reduce_sum(tf.math.abs(cpoints)**2, axis=-1)

    #print("here")
    ######k_fs = _fubini_study_n_potentials(cpoints, t=BASIS['KMODULI'][0])
    #print("here")
    cpoints_stored=[cpoints]
    kappaslist=[kappas]
    #print('degs '+ str(degrees))
    #print('nProjec'+ str(nProjective))
    for i in range(1, nProjective):
        s = tf.reduce_sum(degrees[:i])
        e = s + degrees[i]
        cpoints = points[:, s:e]

        kappas = tf.math.reduce_sum(tf.math.abs(cpoints)**2, axis=-1)

        kappaslist += [kappas]


        cpoints_stored+=[cpoints]
    #print("here2")
    toconcat=[]
    for i in range(0,nProjective):
        #zzbar+=tf.einsum('xi,xj->xij',cpoints_stored[i],tf.math.conj(cpoints_stored[i]))
        X = tf.math.real(cpoints_stored[i])
        Y = tf.math.imag(cpoints_stored[i])

        XX_YY = (tf.einsum('xa,xb->xab',X, X)) + (tf.einsum('xa,xb->xab',Y,Y))
        XY_YX = antisymmetrisemat(tf.einsum('xa,xb->xab',X, Y))
        ones = tf.ones_like(XX_YY[0])
        #zeros= tf.zeros_like(XX_YY[0])
        mask_a = tf.cast(tf.linalg.band_part(ones, 0, -1),dtype=tf.bool) # Upper triangular matrix of 0s and 1s
        mask_b = tf.cast(tf.linalg.band_part(ones, 0, -1)-tf.linalg.band_part(ones, 0, 0),dtype=tf.bool) # Upper triangular matrix of 0s and 1s
 
        upper_triangular_flat_XX_YY = tf.transpose(tf.boolean_mask(tf.transpose(XX_YY), mask_a,axis=0))
        upper_triangular_flat_XY_YX = -tf.transpose(tf.boolean_mask(tf.transpose(XY_YX), mask_b,axis=0))


        toconcat+=[tf.einsum('xi,x->xi',tf.concat([upper_triangular_flat_XX_YY,upper_triangular_flat_XY_YX],axis=-1),kappaslist[i]**(-1))]#XY_YX#tf.concat([ XX_YY,XY_YX], axis=0)

        ##print("here4")
        #realtoadd,imagtoadd = getrealandimagofprod(cpoints_stored[i],return_mat=False)   
        #tempimagsize=int(degrees[i]*(degrees[i]-1)/2)
        #temprealsize=int(degrees[i]*(degrees[i]+1)/2)
        #iterativerealparts=[iterativerealsize*temprealsize,iterativeimagsize*tempimagsize]
        #iterativeimagparts=[iterativerealsize*tempimagsize,iterativeimagsize*temprealsize]
        #iterativereal2 = tf.concat([tf.reshape(tf.einsum('xa,xi->xai', iterativereal, realtoadd),[-1,iterativerealparts[0]]), 
        #                                 - tf.reshape(tf.einsum('xa,xi->xai', iterativeimag, imagtoadd),[-1,iterativerealparts[1]])],axis=-1)#tf.einsum('xa,xi->aijkl', iterativeimag, imagtoadd),[-1,]
        #iterativeimag2 = tf.concat([tf.reshape(tf.einsum('xa,xi->xai', iterativereal, imagtoadd),[-1,iterativeimagparts[0]]), 
        #                                 - tf.reshape(tf.einsum('xa,xi->xai', iterativeimag, realtoadd),[-1,iterativeimagparts[1]])],axis=-1)#tf.einsum('xa,xi->aijkl', iterativeimag, imagtoadd),[-1,]
        #iterativerealsize=iterativerealparts[0]+iterativerealparts[1]
        #iterativeimagsize=iterativeimagparts[0]+iterativeimagparts[1]
        #iterativereal = iterativereal2
        #iterativeimag = iterativeimag2
        ##iterativeimag2 = tf.einsum('aij,akl->aijkl', iterativereal, imagtoadd) + tf.einsum('aij,akl->aijkl', iterativeimag, realtoadd)
    sectimessecbar_over_kappa= tf.concat(toconcat,axis=-1)
    return sectimessecbar_over_kappa

# class SquareDenseVar(tf.keras.layers.Layer):
#     def __init__(self, input_dim, units, activation=tf.square, stddev=0.05,trainable=True,positive_init=True):
#         super(SquareDenseVar, self).__init__()
#         w_init = tf.random_normal_initializer(mean=0.0, stddev=stddev)
#         self.w = tf.Variable(
#             initial_value=tf.math.abs(w_init(shape=(input_dim, units), dtype='float64')) if positive_init else w_init(shape=(input_dim, units), dtype='float64'),
#             trainable=trainable,
#         )
#         self.activation = activation 

#     def call(self, inputs):
#         return self.activation(tf.matmul(inputs, self.w)) 

# class SquareDenseVarNoAct(tf.keras.layers.Layer):
#     def __init__(self, input_dim, units, stddev=0.05,trainable=True,positive_init=True):
#         super(SquareDenseVarNoAct, self).__init__()
#         w_init = tf.random_normal_initializer(mean=0.0, stddev=stddev)
#         self.w = tf.Variable(
#             #initial_value=tf.math.abs(w_init(shape=(input_dim, units), dtype='float64')),
#             initial_value=tf.math.abs(w_init(shape=(input_dim, units), dtype='float64')) if positive_init else w_init(shape=(input_dim, units), dtype='float64'),
#             #initial_value=w_init(shape=(input_dim, units), dtype='float64'),
#             trainable=trainable,
#         )

#     def call(self, inputs):
#         return tf.matmul(inputs, self.w)

# class BiholoModelFuncGENERAL(tf.keras.Model):
#     def __init__(self, layer_sizes,BASIS,activation=tf.square,stddev=0.1,use_zero_network=False):
#         super().__init__()
#         #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
#         set_stddev= stddev#decide init
#         self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
#                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.

#         self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
#         #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1

#         #self.layers_list = [tfk.layers.Dense(units=layer_sizes[i+1],activation=activation)
#         #                    for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
#         #self.layers_list += [tfk.layers.Dense(units=layer_sizes[len(layer_sizes)-1],activation=None)]
#         ##i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
#         final_layer_inits=tf.keras.initializers.Ones if (not use_zero_network) else tf.keras.initializers.Zeros
#         self.layers_list+=[tf.keras.layers.Dense(units=1, use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log
#         self.BASIS=BASIS
#         self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
#         self.ambient=tf.cast(BASIS['AMBIENT'],tf.int32)
#         self.kmoduli=BASIS['KMODULI']
#         if len(self.ambient)==1:
#             print("using single ambient surface bihom func generator")
#             self.bihom_func= bihom_function_generatorQorT(np.array(self.ambient),len(self.ambient),self.kmoduli)
#         else:
#             print("using multi ambient surface bihom func generator")
#             self.bihom_func= bihom_function_generator(np.array(self.ambient),len(self.ambient),self.kmoduli)
#         print("no, using id")
#         self.bihom_func = self.identity
#     def identity(self, inputs):
#         return tf.concat([tf.math.real(inputs), tf.math.imag(inputs)], axis=-1)
    
#     def call(self, inputs):
#         #sum_coords=(tf.reduce_sum(inputs,axis=-1))
#         #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
#         inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
#         #print("ncCoords" +  str(self.nCoords))
#         #norm=tf.math.abs(tf.norm(inputs,axis=-1))
#         inputs =self.bihom_func(inputs)
#         #print(tf.shape(inputs))
#         #print(tf.shape(inputs))
#         #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
#         for layer in self.layers_list[:-1]:
#             inputs = layer(inputs)
#             #print(tf.shape(inputs))
#         #print(len(self.layers_list))
#         ### incorrect!
#         return self.layers_list[-1](inputs)
#         out=(1/np.pi)*(1/2**(len(self.layers_list)))*self.layers_list[-1](tf.math.log(tf.math.abs(inputs)))
#         return tf.clip_by_value(out,-1e6,1e6)

class SquareDenseVar(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation=tf.square, stddev=0.05, trainable=True, positive_init=True):
        super(SquareDenseVar, self).__init__()
        #w_init = tf.random_normal_initializer(mean=0.0, stddev=stddev)
        w_init= tfk.initializers.GlorotUniform()
        initial_value = w_init(shape=(input_dim, units), dtype='float64')
        if positive_init:
            initial_value = tf.math.abs(initial_value)
        self.w = tf.Variable(initial_value, trainable=trainable)
        self.activation = activation 

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

class SquareDenseVarNoAct(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, stddev=0.05, trainable=True, positive_init=True):
        super(SquareDenseVarNoAct, self).__init__()
        #w_init = tf.random_normal_initializer(mean=0.0, stddev=stddev)
        w_init= tfk.initializers.GlorotUniform()
        initial_value = w_init(shape=(input_dim, units), dtype='float64')
        if positive_init:
            initial_value = tf.math.abs(initial_value)
        self.w = tf.Variable(initial_value, trainable=trainable)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

class BiholoModelFuncGENERAL(tf.keras.Model):
    def __init__(self, layer_sizes, BASIS, activation=tf.square, stddev=0.1, use_zero_network=False):
        super().__init__()
        print("activation for biholo model func general: ", activation)
        self.BASIS = BASIS
        self.nCoords = tf.reduce_sum(tf.cast(BASIS['AMBIENT'], tf.int32) + 1)
        self.ambient = tf.cast(BASIS['AMBIENT'], tf.int32)
        self.kmoduli = BASIS['KMODULI']

        self.model = tf.keras.Sequential()
        for i in range(len(layer_sizes) - 2):
            self.model.add(tf.keras.layers.Dense(
                units=layer_sizes[i+1],
                activation=activation))
        
        self.model.add(tf.keras.layers.Dense(
            units=layer_sizes[-1],
            activation=None))

        final_layer_init = tf.keras.initializers.Zeros if use_zero_network else tf.keras.initializers.Ones
        self.model.add(tf.keras.layers.Dense(units=1, use_bias=False, kernel_initializer=final_layer_init))

        # if len(self.ambient) == 1:
        #     print("Using single ambient surface bihom func generator")
        #     self.bihom_func = bihom_function_generatorQorT(np.array(self.ambient), len(self.ambient), self.kmoduli)
        # else:
        #     print("Using multi ambient surface bihom func generator")
        #     self.bihom_func = bihom_function_generator(np.array(self.ambient), len(self.ambient), self.kmoduli)
        print("no, using identity")
        self.bihom_func = self.identity

    def identity(self, inputs):
        return tf.concat([tf.math.real(inputs), tf.math.imag(inputs)], axis=-1)

    def call(self, inputs):
        #inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
        #inputs = self.bihom_func(inputs)
        return self.model(inputs)


# class BiholoModelFuncGENERALforHYMinv(tf.keras.Model):
#     def __init__(self, layer_sizes,BASIS,activation=tf.square,stddev=0.1,use_zero_network=False):
#         super().__init__()
#         #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
#         set_stddev= 0. if use_zero_network else stddev#decide init
#         self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
#                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
#         self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
#         #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
#         self.layers_list+=[tf.keras.layers.Dense(units=layer_sizes[-1], use_bias=False,kernel_initializer=tf.keras.initializers.Ones)]# add the extra free parameter after the log
#         self.BASIS=BASIS
#         self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
#         self.ambient=BASIS['AMBIENT']
#         self.kmoduli=BASIS['KMODULI']
                            
#     def call(self, inputs):
#         #sum_coords=(tf.reduce_sum(inputs,axis=-1))
#         #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
#         inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
#         #print("ncCoords" +  str(self.nCoords))
#         #norm=tf.math.abs(tf.norm(inputs,axis=-1))
#         #inputs =bihomogeneous_section_for_prod(inputs,self.BASIS)
#         inputs =bihomogeneous_section_for_prod(inputs,self.ambient,self.kmoduli)
#         #print(tf.shape(inputs))
#         #print(tf.shape(inputs))
#         #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
#         for layer in self.layers_list[:-1]:
#             inputs = layer(inputs)
#             #print(tf.shape(inputs))
#         #print(len(self.layers_list))
#         ### incorrect!
#         #print("new inv")
#         return  self.layers_list[-1](tf.math.log(tf.math.abs(inputs)))


# class BiholoModelFuncGENERALforHYMinv2(tf.keras.Model):
#     def __init__(self, layer_sizes,BASIS,activation=tf.square,stddev=0.1,use_zero_network=False):
#         super().__init__()
#         #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
#         set_stddev= 0. if use_zero_network else stddev#decide init
#         self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
#                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
#         self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
#         #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
#         self.layers_list+=[tf.keras.layers.Dense(units=1, use_bias=False,kernel_initializer=tf.keras.initializers.Ones)]# add the extra free parameter after the log
#         self.layers_list2 = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
#                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
#         self.layers_list2 += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
#         #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
#         self.layers_list2+=[tf.keras.layers.Dense(units=1, use_bias=False,kernel_initializer=tf.keras.initializers.Ones)]# add the extra free parameter after the log
#         self.BASIS=BASIS
#         self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
                            
#     def call(self, inputs):
#         #sum_coords=(tf.reduce_sum(inputs,axis=-1))
#         #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
#         inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
#         #print("ncCoords" +  str(self.nCoords))
#         #norm=tf.math.abs(tf.norm(inputs,axis=-1))
#         bihom =bihomogeneous_section_for_prod(inputs,self.BASIS)
#         #print(tf.shape(inputs))
#         #print(tf.shape(inputs))
#         #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
#         inputs= bihom
#         #inputs2= bihom
#         for layer in self.layers_list[:-1]:
#             inputs = layer(inputs)
#         for layer in self.layers_list2[:-1]:
#             inputs2 = layer(inputs2)   
#             #print(tf.shape(inputs))
#         #print(len(self.layers_list))
#         ### incorrect!
#         #print("new inv")
#         return  self.layers_list[-1](tf.math.log(inputs))-self.layers_list2[-1](tf.math.log(inputs2))
#         #return  -self.layers_list[-1](tf.math.log(inputs))
    
class BiholoModelFuncGENERALforHYMinv3(tf.keras.Model):
    def __init__(self, layer_sizes,BASIS,activation=tf.square,stddev=0.1,use_zero_network=False,constant_multiplier=1.0):
        super().__init__()
        #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
        #set_stddev= 0. if use_zero_network else stddev#decide init
        # self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
        #                     for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        # self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
        # #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
        # self.layers_list+=[tf.keras.layers.Dense(units=1, use_bias=False,kernel_initializer=tf.keras.initializers.Ones)]# add the extra free parameter after the log
        # self.layers_list2 = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
        #                      for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        # self.layers_list2 += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
        #self.layers_list2+=[tf.keras.layers.Dense(units=1, use_bias=False,kernel_initializer=tf.keras.initializers.Ones)]# add the extra free parameter after the log
        # #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
        final_layer_inits=tf.keras.initializers.Ones if (not use_zero_network) else tf.keras.initializers.Zeros

        self.layers_list = [tf.keras.layers.Dense(units=layer_sizes[i+1],activation=activation)
                            for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        self.layers_list += [tf.keras.layers.Dense(units=layer_sizes[len(layer_sizes)-1],activation=activation)]
        self.layers_list+=[tf.keras.layers.Dense(units=layer_sizes[-1], use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log

        self.layers_list2 = [tf.keras.layers.Dense(units=layer_sizes[i+1],activation=activation)
                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        self.layers_list2 += [tf.keras.layers.Dense(units=layer_sizes[len(layer_sizes)-1],activation=activation)]
        self.layers_list2+=[tf.keras.layers.Dense(units=layer_sizes[-1], use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log
        
        self.BASIS=BASIS
        self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
        self.ambient=tf.cast(BASIS['AMBIENT'],tf.int32)
        self.kmoduli=BASIS['KMODULI']
        if len(self.ambient)==1:
            print("using single ambient surface bihom func generator")
            self.bihom_func= bihom_function_generatorQorT(np.array(self.ambient),len(self.ambient),self.kmoduli)
        else:
            print("using multi ambient surface bihom func generator")
            self.bihom_func= bihom_function_generator(np.array(self.ambient),len(self.ambient),self.kmoduli)
        self.dim_output=layer_sizes[-1]
        self.constant_multiplier=constant_multiplier
                            
    def call(self, inputs):
        #sum_coords=(tf.reduce_sum(inputs,axis=-1))
        #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
        inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
        #print("ncCoords" +  str(self.nCoords))
        #norm=tf.math.abs(tf.norm(inputs,axis=-1))
        bihom =self.bihom_func(inputs)

        #bihom =bihomogeneous_section_for_prod_not_mult(inputs,self.BASIS)
        #print(tf.shape(inputs))
        #print(tf.shape(inputs))
        #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
        inputs= bihom
        inputs2= bihom
        for layer in self.layers_list[:-1]:
            inputs = layer(inputs)
        for layer in self.layers_list2[:-1]:
            inputs2 = layer(inputs2)   
            #print(tf.shape(inputs))
        #print(len(self.layers_list))
        ### incorrect!
        #print("new inv")
        #return  self.layers_list[-1](inputs)/self.layers_list2[-1](inputs2)
        out=self.layers_list[-1](tf.math.log(tf.math.abs(inputs)))-self.layers_list2[-1](tf.math.log(tf.math.abs(inputs2)))
        return self.constant_multiplier*tf.clip_by_value(out,-1e6,1e6)
        #return  -self.layers_list[-1](tf.math.log(inputs))

class BiholoModelFuncGENERALforSigma(tf.keras.Model):
    def __init__(self, layer_sizes,BASIS,linebundleindices,nsections,k_phi,activation=tf.square,stddev=0.1,use_zero_network=False):
        super().__init__()
        #EXPLAIN what ones((2)) is oding?
        indskpModM=[tf.cast(get_monomial_indices(tf.ones((2)),k_phi[i]+tf.math.abs(linebundleindices[i])),tf.int32) for i in range(len(k_phi))]
        indsk=[tf.cast(get_monomial_indices(tf.ones(2),k_phi[i]),tf.int32) for i in range(len(k_phi))] #conj is unnecessary here
        #tf.print(indsk)
        #tf.print(indskpModM)
        self.indslist=(indskpModM,indsk)
        self.k_phi=k_phi
        self.linebundleindices=linebundleindices
        self.nsections=nsections
        final_layer_inits=tf.keras.initializers.Ones if (not use_zero_network) else tf.keras.initializers.Zeros
        #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
        #set_stddev= 0. if use_zero_network else stddev#decide init
        #set_stddev=stddev
        # self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
        #                     for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        # self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
        # #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
        # self.layers_list+=[tf.keras.layers.Dense(units=2*nsections, use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log
        # self.layers_list2 = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
        #                     for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        # self.layers_list2 += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
        # #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
        # self.layers_list2+=[tf.keras.layers.Dense(units=2*nsections, use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log

        self.layers_list = [tf.keras.layers.Dense(units=layer_sizes[i+1],activation=activation)
                            for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        self.layers_list += [tf.keras.layers.Dense(units=layer_sizes[len(layer_sizes)-1],activation=activation)]
        self.layers_list+=[tf.keras.layers.Dense(units=2*nsections, use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log

        self.layers_list2 = [tf.keras.layers.Dense(units=layer_sizes[i+1],activation=activation)
                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
        self.layers_list2 += [tf.keras.layers.Dense(units=layer_sizes[len(layer_sizes)-1],activation=activation)]
        self.layers_list2+=[tf.keras.layers.Dense(units=2*nsections, use_bias=False,kernel_initializer=final_layer_inits)]# add the extra free parameter after the log
        

        self.BASIS=BASIS
        self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
        self.ambient=tf.cast(BASIS['AMBIENT'],tf.int32)
        self.kmoduli=BASIS['KMODULI']
        self.bihom_func= bihom_function_generator(np.array(self.ambient),len(self.ambient),self.kmoduli)
        self.get_deg_kphi_and_mons_class=get_degree_kphiandMmonomials(k_phi,linebundleindices,self.indslist)
        self.get_deg_kphi_and_mons=tf.function(self.get_deg_kphi_and_mons_class.__call__,input_signature=(tf.TensorSpec(shape=[None,self.nCoords], dtype=tf.complex128),))
        #self.get_deg_kphi_and_mons_func=lambda x: get_degree_kphiandMmonomials_func(k_phi,linebundleindices,self.indslist,x)

    def call(self, inputs):
        #sum_coords=(tf.reduce_sum(inputs,axis=-1))
        #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
        inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
        #print("ncCoords" +  str(self.nCoords))
        #norm=tf.math.abs(tf.norm(inputs,axis=-1))
        bihom =self.bihom_func(inputs)
        #tf.print('inp shape')
        #tf.print(tf.shape(inputs))
        sectionsbasis=self.get_deg_kphi_and_mons(inputs)
        #sectionsbasis=get_degree_kphiandMmonomials_func(self.k_phi,self.linebundleindices,self.indslist,inputs)
        #print(sectionsbasis.shape)
        #tf.print('hi')
        #tf.print(sectionsbasis.shape)
        #print(tf.shape(inputs))
        #print(tf.shape(inputs))
        #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
        inputs= bihom
        inputs2= bihom
        for layer in self.layers_list[:-1]:
            inputs = layer(inputs)
        for layer in self.layers_list2[:-1]:
            inputs2 = layer(inputs2)   
            #print(tf.shape(inputs))
        #print(len(self.layers_list))
        ### incorrect!
        #print("new inv")
        to_multiply_sections_real= (self.layers_list[-1](tf.math.log(tf.math.abs(inputs)))-self.layers_list2[-1](tf.math.log(tf.math.abs(inputs2))))
        to_multiply_sections_real = tf.clip_by_value(to_multiply_sections_real,-1e6,1e6)
        to_multiply_sections_complex= tf.complex(to_multiply_sections_real[:,:self.nsections],to_multiply_sections_real[:,self.nsections:])
        #tf.print(to_multiply_sections_complex.shape)#should remove this - why so big?
        #return to_multiply_sections_complex[:,0]
        out=tf.einsum('xi,xi->x',sectionsbasis,to_multiply_sections_complex)
        return out
        #return tf.reduce_sum(to_multiply_sections_complex,axis=-1)#tf.einsum('xi,xi->x',sectionsbasis,to_multiply_sections_complex)

# class BiholoModelFuncGENERALforSectionsSigma(tf.keras.Model):
#     def __init__(self, layer_sizes,BASIS,linebundleindices,activation=tf.square,stddev=0.1,use_zero_network=False):
#         size=question
#         super().__init__()
#         #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
#         set_stddev= 0. if use_zero_network else stddev#decide init
#         self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation)
#                             for i in range(len(layer_sizes)-2-1)]#i.e. 0->1,1->2,... layer_sizes-2->layer_sizes-3->layer_sizes-2. so misses the last 1. this should be 1.
#         self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev)]
#         #i.e. shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1], so the first ones gets up to the +1
#         self.layers_list+=[tf.keras.layers.Dense(units=size, use_bias=False,kernel_initializer=tf.keras.initializers.Ones)]# add the extra free parameter after the log
#         self.BASIS=BASIS
#         self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
                            
#     def call(self, inputs):
#         #sum_coords=(tf.reduce_sum(inputs,axis=-1))
#         #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
#         inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
#         #print("ncCoords" +  str(self.nCoords))
#         #norm=tf.math.abs(tf.norm(inputs,axis=-1))
#         inputs =bihomogeneous_section_for_prod(inputs,self.BASIS)
#         #print(tf.shape(inputs))
#         #print(tf.shape(inputs))
#         #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
#         for layer in self.layers_list[:-1]:
#             inputs = layer(inputs)
#             #print(tf.shape(inputs))
#         #print(len(self.layers_list))
#         ### incorrect!
#         #print("new inv")
#         return  self.layers_list[-1](tf.math.log(tf.math.abs(inputs)))



class BiholoModelFuncGENERALnolog(tf.keras.Model):
    def __init__(self, layer_sizes,BASIS,activation='gelu',stddev=0.1,use_zero_network=False):
        super().__init__() # works in python 3
        #super(BiholoModelFuncGENERALnolog,self).__init__()
        print('activation ' + str(activation))

        #self.layers_list = [tf.keras.layers.Dense(units=size, activation=tf.math.square, use_bias=False)
        set_stddev= 0. if use_zero_network else stddev#decide init
        #self.layers_list = [SquareDenseVar(input_dim=layer_sizes[i],units=layer_sizes[i+1],stddev=set_stddev,activation=activation,positive_init=False)
        #                    for i in range(len(layer_sizes)-2-1)]#i.e. 0,1,... layer_sizes-2->layer_sizes-1. so misses the last 2. these should both be 1.
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=stddev)
        self.layers_list = [tfk.layers.Dense(units=layer_sizes[i+1],kernel_initializer=kernel_initializer,activation=activation)
                            for i in range(len(layer_sizes)-2-1)]#i.e. 0,1,... layer_sizes-2->layer_sizes-1. so misses the last 2. these should both be 1.
        #self.layers_list += [SquareDenseVarNoAct(input_dim=layer_sizes[len(layer_sizes)-2],units=layer_sizes[len(layer_sizes)-1],stddev=set_stddev,positive_init=False)]
        self.layers_list += [tfk.layers.Dense(units=layer_sizes[len(layer_sizes)-1],kernel_initializer=kernel_initializer,activation=None)]
        self.BASIS=BASIS
        self.ambeint=BASIS['AMBIENT']
        self.kmoduli=BASIS['KMODULI']
        self.nCoords=tf.reduce_sum(tf.cast(BASIS['AMBIENT'],tf.int32)+1)
                            
    def call(self, inputs):
        #sum_coords=(tf.reduce_sum(inputs,axis=-1))
        #norm_factor_phase=np.e**((1.j)*tf.cast(tf.math.atan2(tf.math.imag(sum_coords),tf.math.real(sum_coords)),tf.complex128))
        inputs = tf.complex(inputs[:, :self.nCoords], inputs[:, self.nCoords:])
        #print("ncCoords" +  str(self.nCoords))
        #norm=tf.math.abs(tf.norm(inputs,axis=-1))
        #inputs =bihomogeneous_section_for_prod(inputs,self.BASIS)
        inputs =bihomogeneous_section_for_prod(inputs,self.ambient,self.kmoduli)
        #inputs =bihomogeneous_section_for_prod_not_mult(inputs,self.BASIS)
        #print(tf.shape(inputs))
        #print(tf.shape(inputs))
        #return tf.math.log(tf.reduce_sum(inputs,axis=-1))
        for layer in self.layers_list:
            inputs = layer(inputs)
            #print(tf.shape(inputs))
        #print(len(self.layers_list))
        ### incorrect!
        #print("INVERTING ")
        return (inputs)

def make_nn(n_in,n_out,nlayer,nHidden,act='gelu',lastbias=False,use_zero_network=False,kernel_initializer='glorot_uniform'):
   if use_zero_network:
      nn_phi = tf.keras.Sequential()
      nn_phi.add(tfk.Input(shape=(n_in,)))
      for i in range(nlayer):
          nn_phi.add(tfk.layers.Dense(nHidden, activation=act,kernel_initializer='zeros'))
      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
   else:
      nn_phi = tf.keras.Sequential()
      nn_phi.add(tfk.Input(shape=(n_in,)))
      for i in range(nlayer):
          nn_phi.add(tfk.layers.Dense(nHidden, activation=act,kernel_initializer=kernel_initializer))
      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
   return nn_phi
    
