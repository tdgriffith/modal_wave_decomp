import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import matplotlib.axes as axes
import time
import muselsl

# Handy little enum to make code more readable
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 10

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH



if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info, description, sampling frequency, number of channels
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    # Index of the channel (electrode) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    index_channel = [0, 1, 2, 3]
    # Name of our channel for plotting purposes
    ch_names = [ch_names[i] for i in index_channel]
    n_channels = len(index_channel)

    feature_names = utils.get_feature_names(ch_names)

   

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), n_channels))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta, all]
    band_buffer = np.zeros((n_win_test, len(feature_names)))
    #power_buffer=np.zeros((n_win_test, 1))

    # Initilize Plots
    
    plt.ion() 
    fig, axs = plt.subplots(4, 3, figsize=(9, 3), sharey=False)
    
    
    
    

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            #band_powers = utils.compute_band_powers(data_epoch, fs)
            #band_buffer, _ = utils.update_buffer(band_buffer,
                                                #np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            #smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces

            # Alpha Protocol:
            # Simple redout of alpha power, divided by delta waves in order to rule out noise
            #alpha_metric = smooth_band_powers[Band.Alpha] / \
            #    smooth_band_powers[Band.Delta]
            #print('Alpha Relaxation: ', alpha_metric)

            ## PSD for uV timeseries

            winSampleLength, nbCh = data_epoch.shape
            w = np.hamming(winSampleLength)
            dataWinCentered = data_epoch - np.mean(data_epoch, axis=0)  # Remove offset
            dataWinCenteredHam = (dataWinCentered.T * w).T
            dataWinCenteredHam=dataWinCenteredHam**2 #Square for instant power

            NFFT = utils.nextpow2(winSampleLength)
            Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
            PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
           
            f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))
            # SPECTRAL FEATURES
            # Average of band powers
            # Delta <4
            ind_delta, = np.where(f < 4)
            meanDelta = np.mean(PSD[ind_delta, :], axis=0)
            # Theta 4-8
            ind_theta, = np.where((f >= 4) & (f <= 8))
            meanTheta = np.mean(PSD[ind_theta, :], axis=0)
            # Alpha 8-12
            ind_alpha, = np.where((f >= 8) & (f <= 12))
            meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
            # Beta 12-30
            ind_beta, = np.where((f >= 12) & (f < 30))
            meanBeta = np.mean(PSD[ind_beta, :], axis=0)
            # All
            ind_all, = np.where((f < 30))
            meanAll = np.mean(PSD[ind_all, :], axis=0)

            band_powers = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                     meanBeta), axis=0)

            #band_powers = np.log10(band_powers)

            band_buffer, _ = utils.update_buffer(band_buffer,
                                                np.asarray([band_powers]))

                             



            #sorted_list=sorted(((value, index) for index, value in enumerate(PSD)), reverse=True)
            #sorted_list=sorted_list[0:8]
            #sorted_amps=[sorted_list[0][0],sorted_list[1][0],sorted_list[2][0],sorted_list[3][0],sorted_list[4][0],sorted_list[5][0],sorted_list[6][0],sorted_list[7][0]]
            #sorted_index=[sorted_list[0][1],sorted_list[1][1],sorted_list[2][1],sorted_list[3][1],sorted_list[4][1],sorted_list[5][1],sorted_list[6][1],sorted_list[7][1]]
            #sorted_freq=f[sorted_index]
            #print('Top Amplitudes:', np.squeeze(sorted_amps))
            #print('Top frequencies:',sorted_freq)
            
            
            line1,=axs[0,0].plot(np.squeeze(f[ind_delta]),np.squeeze(PSD[ind_delta, 0]),'o')
            line2,=axs[0,0].plot(np.squeeze(f[ind_theta]),np.squeeze(PSD[ind_theta, 0]),'o')
            line3,=axs[0,0].plot(np.squeeze(f[ind_alpha]),np.squeeze(PSD[ind_alpha, 0]),'o')
            line4,=axs[0,0].plot(np.squeeze(f[ind_beta]),np.squeeze(PSD[ind_beta, 0]),'o')
            
            axs[0,0].legend([r'$\delta$ (0-4 hz)',r'$\theta$ (4-8 hz)',r'$\alpha$ (8-12 hz)',r'$\beta$ (12-30 hz)'])
            axs[0,0].set_title(r'Expansion of Instant Bandwidth Power')
            axs[0,0].set_xlabel('Frequency')
            axs[0,0].set_ylabel('Amplitude')

            line5,=axs[1,0].plot(np.squeeze(f[ind_delta]),np.squeeze(PSD[ind_delta, 1]),'o')
            line6,=axs[1,0].plot(np.squeeze(f[ind_theta]),np.squeeze(PSD[ind_theta, 1]),'o')
            line7,=axs[1,0].plot(np.squeeze(f[ind_alpha]),np.squeeze(PSD[ind_alpha, 1]),'o')
            line8,=axs[1,0].plot(np.squeeze(f[ind_beta]),np.squeeze(PSD[ind_beta, 1]),'o')
            
            axs[1,0].legend([r'$\delta$ (0-4 hz)',r'$\theta$ (4-8 hz)',r'$\alpha$ (8-12 hz)',r'$\beta$ (12-30 hz)'])
            axs[1,0].set_title(r'Expansion of Instant Bandwidth Power')
            axs[1,0].set_xlabel('Frequency')
            axs[1,0].set_ylabel('Amplitude')

            
            #axs[0].set_ylim(0,.03)
            #axs[0].set_xlim(0,35)
            

            ## One big bandwidth plot
            #line5,=axs[1].plot(np.squeeze(f),np.squeeze(PSD[:,4]),'o')
            #axs[1].set_ylim(0,.15)
            #axs[1].set_xlabel('Frequency')
            #axs[1].set_ylabel('Amplitude')
            #axs[1].set_title('Power PSD')
            

            # Raw voltage expansion
            line9,=axs[0,1].plot(np.squeeze(f),np.squeeze(PSD[:, 0]))
            #axs[2].set_ylim(0,15)
            axs[0,1].set_xlim(0,55)
            axs[0,1].set_xlabel('Frequency')
            axs[0,1].set_ylabel('Amplitude')
            axs[0,1].set_title('Raw Voltage PSD')

            # Power Time Series
            line10,=axs[0,2].plot(band_buffer[:,0])
            line11,=axs[0,2].plot(band_buffer[:,1])
            line12,=axs[0,2].plot(band_buffer[:,2])
            line13,=axs[0,2].plot(band_buffer[:,3])
            axs[0,2].legend([r'$\delta$ (0-4 hz)',r'$\theta$ (4-8 hz)',r'$\alpha$ (8-12 hz)',r'$\beta$ (12-30 hz)'])



            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(.00001)
            axs[0,1].clear()
            axs[0,2].clear()
            axs[0,0].clear()
            axs[1,0].clear()


            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            # beta_metric = smooth_band_powers[Band.Beta] / \
            #     smooth_band_powers[Band.Theta]
            # print('Beta Concentration: ', beta_metric)

            # Alpha/Theta Protocol:
            # This is another popular neurofeedback metric for stress reduction
            # Higher theta over alpha is supposedly associated with reduced anxiety
            # theta_metric = smooth_band_powers[Band.Theta] / \
            #     smooth_band_powers[Band.Alpha]
            # print('Theta Relaxation: ', theta_metric)

    except KeyboardInterrupt:
        print('Closing!')