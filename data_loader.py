import h5py
import numpy as np
import os

def load_events(file_path:str, filter_polarity:bool=False):
    """
    Input: File path, filter_polarity (bool)

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


def save_h5_events(file_path, events):
    """
    Saves events to a single compound dataset 'events' with fields (x,y,t,p).
    Assuming input 'events' is a list or array with order [t, x, y, p].
    """
    
    # 1. Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 2. Convert to NumPy array if it isn't one already (for efficient slicing)
    events_arr = np.array(events)

    if events_arr.size == 0:
        print(f"Warning: No events to save in {file_path}")
        return

    # 3. Extract columns using correct indices [t, x, y, p]
    #    t = index 0
    #    x = index 1
    #    y = index 2
    #    p = index 3
    t = events_arr[:, 0].astype(np.int64)
    x = events_arr[:, 1].astype(np.int32)
    y = events_arr[:, 2].astype(np.int32)
    p = events_arr[:, 3].astype(np.int8)

    # 4. Create the Compound Data Structure
    #    This matches the specific format your new function was trying to create.
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

    # 5. Save to file
    with h5py.File(file_path, 'w') as f:
        # Create a single dataset named 'events' containing the structured data
        f.create_dataset('events', data=structured, dtype=compound_dtype)
        
    print(f"Saved {len(t)} events to {file_path}")


def height_width(file_path):
    with h5py.File(file_path, 'r') as f:
        node_name = "events"
        ds = f[node_name]
        height = ds['y'][:].max() + 1
        width = ds['x'][:].max() + 1
    return height, width

