import data_loader
import numpy as np

def BAF(sensor_height:int, sensor_width:int, events:np.ndarray, T:float):
    """
    Inputs: sensor dimensions (order: height, width), events array, time threshold T
    
    BAF algorithm implementation for event-based data.
    
    Description: This implementation takes on an batch of events, splits them by polarity and filters them based on the proposed spatiotemporal
    criteria in the BAF paper linked below.

    https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Frame-free+dynamic+digital+vision&btnG=

    Output: Filtered events array, output file path string(This is just for convenience, but will be changed)
    """

    # Initialize timestamp matrix (Both negative and positive) for the filter with added padding for the edges
    timestamp_matrix_pos = np.zeros((sensor_height + 2, sensor_width + 2))
    timestamp_matrix_neg = np.zeros((sensor_height + 2, sensor_width + 2))

    # Initalize output event lists (Both negative and positive)
    output_events = []


    donut = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]


    for event in events:
        t, x, y, p = event
        x = int(x) + 1  # Adjust for padding
        y = int(y) + 1  # Adjust for padding
        p = int(p)
        
        if p == 1:
            if (t - timestamp_matrix_pos[y, x]) <= T:
                output_events.append(event)

            for dy, dx in donut:
                timestamp_matrix_pos[y + dy, x + dx] = t
        
        else:
            if (t - timestamp_matrix_neg[y, x]) <= T:
                output_events.append(event)

            for dy, dx in donut:
                timestamp_matrix_neg[y + dy, x + dx] = t


    
             
    return np.array(output_events), "events/processed_events/BAF/11_4"



PATH = 'events/raw_events/11_4.h5'

# Load events and separate by polarity
events = data_loader.load_events(PATH, filter_polarity=False)

# Load sensor dimensions
sensor_height, sensor_width = data_loader.height_width(PATH)


kaki, name = BAF(sensor_height, sensor_width, events, T=0.3)
data_loader.save_h5_events(name + ".h5", kaki)

