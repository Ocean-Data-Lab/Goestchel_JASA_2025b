# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: 'defaultInterpreterPath: 3.13.5.final.0'
#     language: python
#     name: python3
# ---

# ## Notebook to convert associations

import os
import h5py
import numpy as np
import pickle

# +
with open('../data/associations/Batch1/Baseline/association_2021-11-04_02:00:02.pkl', 'rb') as f:
    
    # Load the association object
    association = pickle.load(f)
#TODO: Convert the .pkl files to HDF5 format and then use the functions form DasOOIDataprocess repo. 

# Explore the keys
print(association.keys())
print(association['assoc_pair'].keys())
print(association['assoc'].keys())
print(len(association['assoc_pair']['north']['hf']))
print(np.shape(association['assoc_pair']['north']['hf'][0]))
print(association['metadata']['north'].keys())

# +
# print(association["assoc_pair"]['north'].keys())
# print(association['metadata'].keys())
# print(association['metadata']['north'].keys())
# print(association['metadata']['assoc_meta'].keys())

print(association['metadata']['north'].items())
print(association['metadata']['south'].items())

# -

def save_assoc(
    filename,
    pair_assoc, pair_loc,
    associations, localizations,
    n_used_hyperbolas, n_rejected_hyperbolas,
    s_used_hyperbolas, s_rejected_hyperbolas,
    n_rejected_list, s_rejected_list,
    n_ds, s_ds,
    dt_kde, bin_width, dt_tol,
    n_shape_x, s_shape_x,
    dt_sel, w_eval, iterations
):
    """Save association results to HDF5 file with compression.
    
    File size reduction: typically 30-70% smaller than pickle.
    Compression: gzip level 4 (good balance of speed/size).
    """
    nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair = pair_assoc
    nhf_pair_loc, nlf_pair_loc, shf_pair_loc, slf_pair_loc = pair_loc
    nhf_associated_list, nlf_associated_list, shf_associated_list, slf_associated_list = associations
    nhf_localizations, nlf_localizations, shf_localizations, slf_localizations = localizations

    with h5py.File(filename, 'w') as f:
        # Helper function to save list of arrays
        def save_list_of_arrays(group, data_list, name):
            """Save a list of numpy arrays as indexed datasets."""
            subgrp = group.create_group(name)
            subgrp.attrs['n_items'] = len(data_list)
            for i, arr in enumerate(data_list):
                if arr is not None and hasattr(arr, 'shape'):
                    if arr.size > 0:
                        subgrp.create_dataset(f'item_{i}', data=arr, compression='gzip', compression_opts=4)
                    else:
                        subgrp.create_dataset(f'item_{i}', data=np.array([]), dtype=arr.dtype)
                else:
                    # Store None as empty dataset with special attribute
                    subgrp.create_dataset(f'item_{i}', data=np.array([]))
                    subgrp[f'item_{i}'].attrs['is_none'] = True
        
        # Save assoc_pair data
        assoc_pair_grp = f.create_group('assoc_pair')
        for cable, hf_data, lf_data in [('north', nhf_assoc_list_pair, nlf_assoc_list_pair),
                                         ('south', shf_assoc_list_pair, slf_assoc_list_pair)]:
            cable_grp = assoc_pair_grp.create_group(cable)
            save_list_of_arrays(cable_grp, hf_data, 'hf')
            save_list_of_arrays(cable_grp, lf_data, 'lf')
        
        # Save pair_loc data
        pair_loc_grp = f.create_group('pair_loc')
        for cable, hf_data, lf_data in [('north', nhf_pair_loc, nlf_pair_loc),
                                         ('south', shf_pair_loc, slf_pair_loc)]:
            cable_grp = pair_loc_grp.create_group(cable)
            save_list_of_arrays(cable_grp, hf_data, 'hf')
            save_list_of_arrays(cable_grp, lf_data, 'lf')
        
        # Save assoc data
        assoc_grp = f.create_group('assoc')
        for cable, hf_data, lf_data in [('north', nhf_associated_list, nlf_associated_list),
                                         ('south', shf_associated_list, slf_associated_list)]:
            cable_grp = assoc_grp.create_group(cable)
            save_list_of_arrays(cable_grp, hf_data, 'hf')
            save_list_of_arrays(cable_grp, lf_data, 'lf')
        
        # Save localizations data
        loc_grp = f.create_group('localizations')
        for cable, hf_data, lf_data in [('north', nhf_localizations, nlf_localizations),
                                         ('south', shf_localizations, slf_localizations)]:
            cable_grp = loc_grp.create_group(cable)
            save_list_of_arrays(cable_grp, hf_data, 'hf')
            save_list_of_arrays(cable_grp, lf_data, 'lf')
        
        # Save hyperbolas
        hyp_grp = f.create_group('hyperbolas')
        for cable, used, rejected in [('north', n_used_hyperbolas, n_rejected_hyperbolas),
                                       ('south', s_used_hyperbolas, s_rejected_hyperbolas)]:
            cable_grp = hyp_grp.create_group(cable)
            save_list_of_arrays(cable_grp, used, 'used')
            save_list_of_arrays(cable_grp, rejected, 'rejected')
        
        # Save rejected lists
        rej_grp = f.create_group('rejected')
        save_list_of_arrays(rej_grp, n_rejected_list, 'north')
        save_list_of_arrays(rej_grp, s_rejected_list, 'south')
        
        # Save metadata
        meta_grp = f.create_group('metadata')
        
        # Save dataset attributes
        north_meta = meta_grp.create_group('north')
        if n_ds is not None and hasattr(n_ds, 'attrs'):
            for key, value in n_ds.attrs.items():
                try:
                    north_meta.attrs[key] = value
                except (TypeError, ValueError):
                    # If value can't be stored as HDF5 attribute, convert to string
                    north_meta.attrs[key] = str(value)
        else:
            for key, value in n_ds.items():
                try:
                    north_meta.attrs[key] = value
                except (TypeError, ValueError):
                    # If value can't be stored as HDF5 attribute, convert to string
                    north_meta.attrs[key] = str(value)
        
        south_meta = meta_grp.create_group('south')
        if s_ds is not None and hasattr(s_ds, 'attrs'):
            for key, value in s_ds.attrs.items():
                try:
                    south_meta.attrs[key] = value
                except (TypeError, ValueError):
                    # If value can't be stored as HDF5 attribute, convert to string
                    south_meta.attrs[key] = str(value)
        else:
            for key, value in s_ds.items():
                try:
                    south_meta.attrs[key] = value
                except (TypeError, ValueError):
                    # If value can't be stored as HDF5 attribute, convert to string
                    south_meta.attrs[key] = str(value)
                    
        # Save association metadata
        assoc_meta = meta_grp.create_group('assoc_meta')
        assoc_meta.attrs['dt_kde'] = dt_kde
        assoc_meta.attrs['bin_width'] = bin_width
        assoc_meta.attrs['dt_tol'] = dt_tol
        assoc_meta.attrs['n_shape_x'] = n_shape_x
        assoc_meta.attrs['s_shape_x'] = s_shape_x
        assoc_meta.attrs['dt_sel'] = dt_sel
        assoc_meta.attrs['w_eval'] = w_eval
        assoc_meta.attrs['iterations'] = iterations



# +
assoc_dir = '../data/associations/Batch5'
batch_dirs = os.listdir(assoc_dir)
for settings in batch_dirs: 
    for file in os.listdir(os.path.join(assoc_dir, settings)):
        if file.endswith('.pkl'):
            with open(os.path.join(assoc_dir, settings, file), 'rb') as f:
                association = pickle.load(f)
                pair_assoc = association['assoc_pair']
                pair_loc = association['pair_loc']
                assoc = association['assoc']
                localizations = association['localizations']
                save_assoc(
                    filename=os.path.join(assoc_dir, settings, file.replace('.pkl', '.h5')),
                    pair_assoc=(pair_assoc['north']['hf'], pair_assoc['north']['lf'],
                                pair_assoc['south']['hf'], pair_assoc['south']['lf']),
                    pair_loc=(pair_loc['north']['hf'], pair_loc['north']['lf'],
                              pair_loc['south']['hf'], pair_loc['south']['lf']),
                    associations=(assoc['north']['hf'], assoc['north']['lf'],
                                  assoc['south']['hf'], assoc['south']['lf']),
                    localizations=(localizations['north']['hf'], localizations['north']['lf'],
                                   localizations['south']['hf'], localizations['south']['lf']),
                    n_used_hyperbolas=association['hyperbolas']['north']['used'],
                    n_rejected_hyperbolas=association['hyperbolas']['north']['rejected'],
                    s_used_hyperbolas=association['hyperbolas']['south']['used'],
                    s_rejected_hyperbolas=association['hyperbolas']['south']['rejected'],
                    n_rejected_list=association['rejected']['north'],
                    s_rejected_list=association['rejected']['south'],
                    n_ds=association['metadata']['north'],
                    s_ds=association['metadata']['south'],
                    dt_kde=association['metadata']['assoc_meta'].get('dt_kde', None),
                    bin_width=association['metadata']['assoc_meta'].get('bin_width', None),
                    dt_tol=association['metadata']['assoc_meta'].get('dt_tol', None),
                    n_shape_x=association['metadata']['assoc_meta'].get('n_shape_x', None),
                    s_shape_x=association['metadata']['assoc_meta'].get('s_shape_x', None),
                    dt_sel=association['metadata']['assoc_meta'].get('dt_sel', None),
                    w_eval=association['metadata']['assoc_meta'].get('w_eval', None),
                    iterations=association['metadata']['assoc_meta'].get('iterations', None)
                )
                
                

# -

def load_assoc(filename):
    """Load association results from HDF5 file.
    
    Returns
    -------
    dict
        Dictionary with the same structure as saved by save_assoc().
    """
    def load_list_of_arrays(group):
        """Load a list of numpy arrays from indexed datasets."""
        n_items = group.attrs['n_items']
        result = []
        for i in range(n_items):
            dataset = group[f'item_{i}']
            if dataset.attrs.get('is_none', False):
                result.append(None)
            else:
                result.append(dataset[:])
        return result
    
    with h5py.File(filename, 'r') as f:
        # Load assoc_pair
        assoc_pair = {
            'north': {
                'hf': load_list_of_arrays(f['assoc_pair/north/hf']),
                'lf': load_list_of_arrays(f['assoc_pair/north/lf'])
            },
            'south': {
                'hf': load_list_of_arrays(f['assoc_pair/south/hf']),
                'lf': load_list_of_arrays(f['assoc_pair/south/lf'])
            }
        }
        
        # Load pair_loc
        pair_loc = {
            'north': {
                'hf': load_list_of_arrays(f['pair_loc/north/hf']),
                'lf': load_list_of_arrays(f['pair_loc/north/lf'])
            },
            'south': {
                'hf': load_list_of_arrays(f['pair_loc/south/hf']),
                'lf': load_list_of_arrays(f['pair_loc/south/lf'])
            }
        }
        
        # Load assoc
        assoc = {
            'north': {
                'hf': load_list_of_arrays(f['assoc/north/hf']),
                'lf': load_list_of_arrays(f['assoc/north/lf'])
            },
            'south': {
                'hf': load_list_of_arrays(f['assoc/south/hf']),
                'lf': load_list_of_arrays(f['assoc/south/lf'])
            }
        }
        
        # Load localizations
        localizations = {
            'north': {
                'hf': load_list_of_arrays(f['localizations/north/hf']),
                'lf': load_list_of_arrays(f['localizations/north/lf'])
            },
            'south': {
                'hf': load_list_of_arrays(f['localizations/south/hf']),
                'lf': load_list_of_arrays(f['localizations/south/lf'])
            }
        }
        
        # Load hyperbolas
        hyperbolas = {
            'north': {
                'used': load_list_of_arrays(f['hyperbolas/north/used']),
                'rejected': load_list_of_arrays(f['hyperbolas/north/rejected'])
            },
            'south': {
                'used': load_list_of_arrays(f['hyperbolas/south/used']),
                'rejected': load_list_of_arrays(f['hyperbolas/south/rejected'])
            }
        }
        
        # Load rejected
        rejected = {
            'north': load_list_of_arrays(f['rejected/north']),
            'south': load_list_of_arrays(f['rejected/south'])
        }
        
        # Load metadata
        metadata = {
            'north': dict(f['metadata/north'].attrs),
            'south': dict(f['metadata/south'].attrs),
            'assoc_meta': dict(f['metadata/assoc_meta'].attrs)
        }
        
        return {
            'assoc_pair': assoc_pair,
            'pair_loc': pair_loc,
            'assoc': assoc,
            'localizations': localizations,
            'hyperbolas': hyperbolas,
            'rejected': rejected,
            'metadata': metadata
        }


# +
test_assoc = load_assoc('../data/associations/Batch1/Baseline/association_2021-11-04_02:00:02.h5')
print(test_assoc['assoc_pair']['north']['hf'])

test2 = h5py.File('../data/associations/Batch1/Baseline/association_2021-11-04_02:00:02.h5', 'r')

print(test2['assoc_pair/north/hf/item_0/'][:])  # Access the first item in the north hf list
