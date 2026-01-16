import h5py
import numpy as np
import os

def load_events(file_path:str, filter_polarity:bool=False):
    """
    Input: File path for original .h5 file, Split by polarity 

    Loads .h5 file events into an array with columns (t, x, y, p).

    Output: events (np.ndarray of shape [N,4])   OR   pos, neg (two np.ndarrays of shape [N,4], split by polarity)
    """
    with h5py.File(file_path, 'r') as f:
        node_name = "events"
        ds = f[node_name]

        x = ds['x'][:]
        y = ds['y'][:]
        t = ds['t'][:] * 1e-3 # convert from microseconds to seconds
        p = ds['p'][:] 

        events = np.stack([t, x, y, p], axis=1)

        if filter_polarity:
            pos = events[events[:, 3] == 1]
            neg = events[events[:, 3] == 0]

            return pos, neg

    return events




def save_h5_events(file_path:str, events:np.ndarray):
    """
    Input: File patch for saved .h5 file, Events array

    Saves events [x, y, t, p] to a single compound dataset 'events' in a .h5 file.
    
    Output: HD file at specified path
    """
    
    # Check for directory
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load columns
    t = events[:, 0].astype(np.int64)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    p = events[:, 3].astype(np.int8)

    # Create the Compound Data Structure, to match the original format
    compound_dtype = np.dtype([
        ('x', np.int32),
        ('y', np.int32),
        ('t', np.int64),
        ('p', np.int8),
    ])

    structured = np.empty(len(t), dtype=compound_dtype)
    structured['x'] = x
    structured['y'] = y
    structured['t'] = t
    structured['p'] = p

    # Save to .h5
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('events', data=structured, dtype=compound_dtype)
        
    print(f"Saved {len(t)} events to {file_path}")



def height_width(file_path):
    with h5py.File(file_path, 'r') as f:
        node_name = "events"
        ds = f[node_name]
        height = ds['y'][:].max() + 1
        width = ds['x'][:].max() + 1
    return height, width

