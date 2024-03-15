# =============================================================================
#  Authors
# =============================================================================
# Name: Olivier PANICO
# corresponding author: olivier.panico@free.fr

# =============================================================================
#  Imports
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

### MATPLOTLIB PARAMS ###
plt.rcParams.update({'font.size': 16, 'figure.figsize': [8, 5],})
plt.rcParams.update({'xtick.top':True, 'xtick.bottom':True, 
                    'xtick.major.width':2, 'xtick.minor.width':2, 
                    'xtick.major.size':6, 'xtick.minor.size':3, 
                    'xtick.color':'black', 'xtick.direction':'in'})
plt.rcParams.update({'ytick.left':True, 'ytick.right':True, 
                    'ytick.major.width':2, 'ytick.minor.width':2, 
                    'ytick.major.size':6, 'ytick.minor.size':3, 
                    'ytick.color':'black', 'ytick.direction':'in'})


# =============================================================================
#  Remark
# =============================================================================
# Utility function for plotting



clist_jet = plt.cm.jet

def get_cmap_list(n, cmap_name, verbose=0):
    colors_ind = np.linspace(0,1,n)
    
    if cmap_name=='viridis':
        cmap=plt.cm.viridis
    elif cmap_name=='bwr':
        cmap=plt.cm.bwr
    else:
        if verbose==1:
            print('choosing default cmap jet')
        cmap=plt.cm.jet

    return cmap(colors_ind)



def plot_1d(ydata, xdata=None, fig=None, ax=None, grid=False, constrained_layout=False, clist='jet', xlabel=None, ylabel=None, xscale=None, yscale=None, **kwargs):
    
    if type(ydata) is list:
        ydata=np.array(ydata)

    assert ydata.ndim == 1 or ydata.ndim == 2
    if ydata.ndim == 2:
        nt,nx = np.shape(ydata)
        colors_ind = np.linspace(0,1,nx)
        if fig is None or ax is None:
            fig, ax = plt.subplots(constrained_layout=constrained_layout)
        for i in range(nx):
            if xdata is None:
                if clist=='jet':
                    ax.plot(ydata[:,i], color=clist_jet(colors_ind[i]), **kwargs)
                else:
                    ax.plot(ydata[:,i], **kwargs)
            else:
                if clist=='jet':
                     ax.plot(xdata,ydata[:,i], color=clist_jet(colors_ind[i]), **kwargs)
                else:
                    ax.plot(xdata,ydata[:,i], **kwargs)
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(constrained_layout=constrained_layout)
        if xdata is None:
            ax.plot(ydata, **kwargs)
        else:
            ax.plot(xdata,ydata, **kwargs)
        
    if xscale=='log':
        ax.set_xscale('log')
    if yscale=='log':
        ax.set_yscale('log')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid:
        ax.tick_params(which='major',axis='both', gridOn=True)
        ax.tick_params(which='minor',axis='both', gridOn=True, grid_linestyle='dotted')

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    if fig is None or ax is None and constrained_layout is None:
        print('using tight_layout in plot_1d')
        plt.tight_layout()
    
    return fig, ax
    


def plot_2d(data, xdata=None, ydata=None,fig = None, ax=None, cbar=True, constrained_layout=False, xlabel=None, ylabel=None, xscale=None, yscale=None, **kwargs):
    
    if kwargs.get('cmap') is None:
        kwargs['cmap'] = 'jet'

    if fig is None and ax is None:
        fig, ax = plt.subplots(constrained_layout=constrained_layout)
    
    if xdata is not None and ydata is not None:
        im = ax.pcolormesh(xdata, ydata, data, shading='auto', **kwargs)
    else:
        im = ax.pcolormesh(data, shading='auto', **kwargs)
    
    if cbar:
        # ~ plt.colorbar(spacing='proportional',drawedges=True)
        # ~ fig.colorbar(im, ax=ax)
        plt.colorbar(im, ax=ax, format='%.e')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
      
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    if fig is None or ax is None:
        plt.tight_layout()

    return fig, ax, im


def prep_multiple_subplots(nrows, ncols, figsize=None, axgrid=None, constrained_layout=True, sharex=False, sharey=False, **kwargs):
    
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey, constrained_layout=constrained_layout, **kwargs)
    
    for i,ax in enumerate(fig.get_axes()):
        if i in axgrid:
            ax.tick_params(which='major',axis='both', gridOn=True)
            ax.tick_params(which='minor',axis='both', gridOn=True, grid_linestyle='dotted')

    
    # ~ plt.tight_layout()
    
    return fig, axs


