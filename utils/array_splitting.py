# =============================================================================
#  Authors
# =============================================================================
# Name: Olivier PANICO
# corresponding author: olivier.panico@free.fr

# =============================================================================
#  Imports
# =============================================================================
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast   #Used to chunk an array into overlapping segments (can mess with memory buffer)

# =============================================================================
#  Remark
# =============================================================================
# Several function for splitting arrays
# Always prefer split_array if no overlap is needed
# If overlap is needed prefer custom split except for large arrays then use chunk_data

# =============================================================================
#  Main functions    
# =============================================================================

def custom_split_1d(A, nperseg, noverlap=0, zero_padding=False):
    '''
    Complexity n
    Custom split into overlapping segments using lists 
    Similar to np.split but authorize the use of overlapping segments
    
    A: array
        input 1d array to split
    nperseg: int
        number of points per segment
    noverlap: int (optional)
        number of overlapping points per segments
    zero_padding: bool (optional)
        if True and if the last segment is not complete will complete with zeros
    
    return the splitted array B of shape (nbseg, nperseg)
    '''
    n = len(A)
    #number of segments that the split will produce
    nbseg = (n - nperseg) // (nperseg-noverlap) + 1
    #number of values that are in the incomplete last segment
    #equal to zero if the decomposition is exact 
    noverhang = n - (nbseg*nperseg - (nbseg-1)*noverlap)

    #list where the segments will be filled
    result = []
    #index with steps: nperseg-noverlap
    index = 0
    
    #add new segment until it reaches the last one
    while index + nperseg <= n:
        segment = A[index:index + nperseg] #each segment of size nperseg
        
        result.append(segment)
        index += nperseg - noverlap #the step is nperseg-noverlap

    #If there is an overhang and zero_padding is true then will include the last window completed with zeros
    if noverhang>0 and zero_padding is True:
        last_segment = np.zeros((nperseg)) 
        # print('last seg: ', (last_segment))
        # print('noverhang = ', noverhang)
        # print(' last seg until 0 = ', (last_segment[:noverhang]))
        # print(' A[index:]', (A[index:]))
        
        len_last_values = len(A[index:])
        last_segment[:len_last_values] = A[index:] #add the last values to a segment filled with zeros
        
        result.append(last_segment)

    return np.array(result)


def custom_split_2d(A, nperseg, noverlap=0, axis=0, zero_padding=False):
    '''
    Complexity n**2
    input:
        A: array
            input array of shape (n,p)
    
    return
        B: array
        splitted array of shape (nbseg, nperseg, p) if decomposition on first axis
                                (nbseg, n, nperseg) if decomposition on second axis
    '''
    n,p = np.shape(A)

    if axis==0:
        nbseg = (n - nperseg) // (nperseg-noverlap) + 1 #Calculate the number of segments
        noverhang = n - (nbseg*nperseg - (nbseg-1)*noverlap) #number of overlapping values
        if noverhang>0 and zero_padding is True:
            B = np.zeros((nbseg+1, nperseg, p)) #if zero_padding will add one new window
        else:
            B = np.zeros((nbseg, nperseg, p)) #Else just needs the normal amount of windows
        for i in range(p):
            result = custom_split_1d(A[:,i], nperseg, noverlap, zero_padding) #Call the 1d function on each slice

            B[:,:, i] = result 
            
    elif axis==1:

        # Other method that works correctly but instead of resulting with an array of shape: (nbseg, n, nperseg) gives shape (nbseg, nperseg, n)
        # it would require the correct transposition but i'm not familiar enough with np array subsitutions
        # return (custom_split_2d(np.transpose(A), nperseg, noverlap, axis=0, zero_padding=zero_padding)) 
        
        #classic method
        nbseg = (p - nperseg) // (nperseg-noverlap) + 1
        noverhang = p - (nbseg*nperseg - (nbseg-1)*noverlap)
        if noverhang>0 and zero_padding is True:
            B = np.zeros((nbseg+1, n, nperseg))
        else:
            B = np.zeros((nbseg, n, nperseg))
        for i in range(n):
            result = custom_split_1d(A[i,:], nperseg, noverlap, zero_padding)
            B[:,i,:] = result
            
    return B


def split_array_1d(A, nperseg, zero_padding=False):
    '''
    Split a 1d array on segments of nperseg pts
    If the decomposition is not exact, will cast away the last segment
    unless zero_padding =True, then will fill it with zeros
    '''
    n = len(A)
    nbseg = int(n/nperseg)

    #If the last segment is not complete
    #print(nbseg, nperseg, nt)
    if (nbseg*nperseg)<n:
        #If zero padding then we fill the rest of the last segment with 0
        if zero_padding:
            B = np.zeros((nperseg*(nbseg+1)))
            B[:n] = A
        #If not zero padding then we drop the last segment because incomplete
        else:
            B = np.zeros((nperseg*nbseg))
            B = A[:nbseg*nperseg] 
    else:
        B = np.zeros((nperseg*nbseg))
        B = A
    
    splitted_sig = np.split(B,nbseg)
    splitted_sig = np.array(splitted_sig)
    
    return splitted_sig


def split_array_2d(A, nperseg, zero_padding=False):
    '''
    Split an array into segments on a third dimension
    For now noverlap is not functional 
    '''
    nt,nx = np.shape(A)
    #Calculate the number of overlapping segments
    nbseg = int(nt/nperseg)
    #If the last segment is not complete
    #print(nbseg, nperseg, nt)
    if (nbseg*nperseg)<nt:
        #If zero padding then we fill the rest of the last segment with 0
        if zero_padding:
            B = np.zeros((nperseg*(nbseg+1), nx))
            B[:nt, nx] = A
        #If not zero padding then we drop the last segment because incomplete
        else:
            B = np.zeros((nperseg*nbseg, nx))
            B = A[:nbseg*nperseg,:] 
    else:
        B = np.zeros((nperseg*nbseg, nx))
        B = A[:,:]
    
    splitted_sig = np.split(B,nbseg)
    splitted_sig = np.array(splitted_sig)
    
    return splitted_sig



###
### To be handled with care
### 

def chunk_data(data,nperseg,noverlap=0, zero_padding = False, flatten_inside_window=True):
    '''
    !!! CAREFUL FUNCTION HAS TO BE HANDLED WITH CARE (can produce errors when calculations are performed with the chunked array due to memory conflicts)
                                                      => When possible (no overlapping) use np.split 
    !!!
    
    This function is used to chunk an array into windows of size nperseg with an overlapping size of noverlap.
    If the last window is not full, it can pad the last empty spaces with zeros
    
    
    Parameters
    ----------
    data : array
        The array we want to slice into pieces.
    nperseg : int
        size of the windows (number of points).
    noverlap : int, optional
        size of the overlap (number of points). The default is 0.
    zero_padding ; bool, optional
        if True and if the last window is not complete, will add 0 values at the end
    flatten_inside_window : bool, optional
        Reduit le nombre de dimension du tableau final. The default is True.

    Returns
    -------
    narray
        Sliced array.

    '''

    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    nbseg = (data.shape[0] - nperseg) // (nperseg - noverlap) + 1
    # overhang is the extra values that don't fit inside numwindows
    overhang = data.shape[0] - (nbseg*nperseg - (nbseg-1)*noverlap)

    
    # if there's overhang, need an extra window and a zero pad on the data
    if overhang != 0 and zero_padding:
            nbseg += 1
            newdata = np.zeros((nbseg*nperseg - (nbseg-1)*noverlap,data.shape[1]))
            newdata[:data.shape[0]] = data
            data = newdata
        
    # Decomposition 
    ### Careful: decomposition with as_strided works fine for read only but hazardous for calculations
    sz = data.dtype.itemsize
    ret = ast(
            data,
            shape=(nbseg,nperseg*data.shape[1]),
            strides=((nperseg-noverlap)*data.shape[1]*sz,sz),
            writeable=False
            )

    # We make a copy of the readonly array to not mess with memory
    copyret = ret.copy()

    if flatten_inside_window:
        return copyret
    else:
        return copyret.reshape((nbseg,-1,data.shape[1]))
    
    

### Sanity checks ###
# custom split should be equal to  split_1d_array if noverlap = 0
# chunk data should be equal to custom split in every configurations

def sanity_check():
    import time
    n = 1000
    p = 5000
    
    print('### create array of :', n*p, ' points ###')
    te = np.zeros((n,p))
    count=0
    for i in range(n):
        for j in range(p):
            te[i,j] = count
            count+=1
    
    print('### test with no overlap ###')
    tinit = time.time()
    a = custom_split_2d(te, nperseg = 256, noverlap = 0)
    print("time custom split (a): ", time.time()-tinit)
    tinit=time.time()
    b = split_array_2d(te, nperseg = 256)
    print("time np split (b): ", time.time()-tinit)
    tinit=time.time()
    c = chunk_data(te,nperseg = 256, noverlap = 0, flatten_inside_window=False)
    print("time chunk data (c): ", time.time()-tinit)
    
    print("is (a-b)==0", not np.any(a-b))
    print("is (a-c)==0", not np.any(a-c))
    
    print('### test with overlap ###')
    tinit = time.time()
    a = custom_split_2d(te, nperseg = 256, noverlap = 128)
    print("time custom split (a): ", time.time()-tinit)
    tinit=time.time()
    c = chunk_data(te,nperseg = 256, noverlap = 128, flatten_inside_window=False)
    print("time chunk data (c): ", time.time()-tinit)
    
    print("is (a-c)==0", not np.any(a-c))
    
    
    