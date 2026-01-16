import h5py
import numpy as np
import cv2
import argparse



def visualize_hdf5(file_path, events_per_frame, resolution=(692, 520)):
    with h5py.File(file_path, 'r') as f:
        x = f['events']['x'][:]
        y = f['events']['y'][:]
        t = f['events']['t'][:]
        p = f['events']['p'][:]  # polarity

    width, height = resolution
    canvas = np.zeros((height, width, 3), dtype=np.uint8)


    print(f"Loaded {len(x)} events. Displaying...")
    for i in range(0, len(x), events_per_frame):
        canvas[:] = 0
        end = min(i + events_per_frame, len(x))
        for j in range(i, end):
            color = (0, 255, 0) if p[j] else (0, 0, 255)
            if 0 <= x[j] < width and 0 <= y[j] < height:
                canvas[y[j], x[j]] = color

        cv2.imshow("Event Stream", canvas)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DVS HDF5 eye-tracking data.")
    parser.add_argument("file", help="Path to the .hdf5 file")
    parser.add_argument("--events", type=int, default=2012, help="Events per frame")
    parser.add_argument("--width", type=int, default=692, help="Sensor width")
    parser.add_argument("--height", type=int, default=520, help="Sensor height")
    args = parser.parse_args()

    visualize_hdf5(args.file, events_per_frame=args.events, resolution=(args.width, args.height))
