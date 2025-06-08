def f (path=None,Nmics = 64,Nsamp = 200,C=1,x_min=0,x_max=8,x_margin=1,x_resolution=200,y_min=-8,y_max=8,y_margin=1,y_resolution=200,path_mode=False,obstacles=[(3, -1), (2, 2), (1, -3)]):
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the data from the text file
    if path!=None:
        data = np.loadtxt(path)

        # Check the total number of values
        total_values = data.size
        expected_values = 64 * 200

        # Ensure the total values can be reshaped into the desired shape
        if total_values == expected_values:
            reshaped_data = data.reshape((64, 200))
        else:
            print("Total values do not match expected shape. Found:", total_values)

    # Main system parameters


    # Source: coordinates
    src = (0, 0)

    # Spacing between microphones
    pitch = 0.1

    # Proxy for sampling rate
    dist_per_samp = 0.1


    # Time dilation factor for sinc pulse: how narrow
    SincP = 5.0

    # Locations of microphones
    mics = []
    if Nmics % 2 == 0:
        count = Nmics // 2
        while count:
            mics.append((0.0, count * pitch))
            mics.append((0.0, -count * pitch))
            count -= 1


    # Define the time basis for computing samples
    t = np.linspace(0, (Nsamp - 1) * dist_per_samp, Nsamp)

    # Source sound wave - sinc wave with narrowness determined by parameter
    def wsrc(t, SincP=5):
        return np.sinc(SincP * t)

    # Distance from src to a mic after reflecting through pt
    def dist(src, pt, mic):
        # Distance from source to point obstacle
        d1 = np.sqrt((src[0] - pt[0])**2 + (src[1] - pt[1])**2)
        
        # Distance from point obstacle to microphone
        d2 = np.sqrt((pt[0] - mic[0])**2 + (pt[1] - mic[1])**2)
        
        return d1 + d2

    # Generate microphone outputs with reflections from all obstacles
    def generate_mic_outputs():
        mic_outputs = np.zeros((Nmics, Nsamp))
        for n in range(Nmics):
            mic_pos = mics[n]
            for obstacle in obstacles:
                # Calculate the total distance
                total_distance = dist(src, obstacle, mic_pos)

                # Calculate the time delay in samples
                delay_samples = int(total_distance / (C * dist_per_samp))
                
                # Fill the mic output with the delayed signal
                if delay_samples < Nsamp:
                    mic_outputs[n, delay_samples:] += wsrc(t[:Nsamp - delay_samples])
        
        return mic_outputs

    mic_outputs = generate_mic_outputs()

    # Simulation setup for delay-and-sum
    x_values = np.linspace(x_min, x_max, x_resolution)  # X-axis for obstacle positions
    y_values = np.linspace(y_min, y_max, y_resolution)  # Y-axis for mic positions

    def delay_and_sum(mic_outputs):
        # # Create a grid for reconstruction
        # x_values = np.linspace(-2, 5, 100)  # X-axis for obstacle positions
        # y_values = np.linspace(-3, 3, Nmics)  # Y-axis for mic positions
        reconstructed_image = np.zeros((len(y_values), len(x_values)))
        
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                # Calculate distance from the source to (x, y) and then to each mic
                for n in range(Nmics):
                    mic_pos = mics[n]
                    # Calculating the total distance
                    total_distance = dist(src, (x, y), mic_pos)
                    
                    # Calculate time delay in samples
                    delay_samples = int(total_distance / (C * dist_per_samp))
                    
                    if delay_samples < Nsamp:
                        reconstructed_image[j, i] += mic_outputs[n, delay_samples]

        return reconstructed_image
    if path_mode==False:
        # reconstructed_image = delay_and_sum(mic_outputs)
        reconstructed_image = delay_and_sum(mic_outputs)
    else:
        reconstructed_image = delay_and_sum(reshaped_data)

    def plot_reconstructed_image(reconstructed_image):


        # Calculate the extent dynamically to cover all obstacles
        extent_x = (x_min - x_margin, x_max + x_margin)
        extent_y = (y_min - y_margin, y_max + y_margin)

        # Plot the reconstructed image
        plt.figure(figsize=(10, 6))
        plt.imshow(reconstructed_image, extent=(extent_x[0], extent_x[1], extent_y[0], extent_y[1]), aspect='auto', origin='lower')
        plt.colorbar(label='Amplitude')
        plt.title('Reconstructed Image from Delay-and-Sum Algorithm')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.show()

    plot_reconstructed_image(reconstructed_image)


f(path=r"t.txt",C=1,Nmics = 64,Nsamp = 200,x_min=0,x_max=8,x_margin=1,x_resolution=200,y_min=-8,y_max=8,y_margin=1,y_resolution=200,path_mode=False,obstacles=[(3, -1), (2, 2), (1, -3)])

