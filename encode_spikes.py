import numpy as np

def encode_spikes(spike_frequency, time_duration, time_counter, time_step=0.001, verbose=False, plots=False):
    ## CCA 23 milisekund

    times = np.arange(0, durationS, timeStepS)  # [0:timeStepS:durationS]	% a vector with each time step
    finalout = np.zeros(int(1 / timeStepS))
    vt = np.random.rand(len(times))

    spikes = (spike_frequency * timeStepS) > vt
    spikes = spikes.astype(int)  # spikes sa pouzije ako event.p
    # for i,spike in enumerate(spikes):
    #  if spike:finalout[i]=1
    spikeTimes = np.asarray(np.where(spikes))  # Spike times, pouzije sa ako event.t
    spikeTimes = spikeTimes.astype(int)

    # spikeTimes+=time_counter  !!!!!!!!!!!!!

    # finalSpikes = np.ones(spikeTimes.shape).astype(int)

    if verbose:
        print(spikes)
        print("number of spikes: ", np.sum(spikes))
        # print(finalout)
        # print(sum(finalout))
        print(spikeTimes)
        # plt.figure()
        # plt.plot(finalout)

        spikeIntervals = spikeTimes[0]  # -spikeTimes[:len(spikeTimes)-1]
        spikeIntervals = spikeIntervals[1:len(spikeIntervals)] - spikeIntervals[:len(spikeIntervals) - 1]
        print(spikeIntervals)
    # print(spikeIntervals2)

    if plots:
        binSize = 1

        x = np.arange(0, 99, binSize)

        fig1 = plt.figure()

        # exp = expon.pdf(1 / (spikesPerS * timeStepS),x)
        # plt.plot(exp)

        intervalDist = plt.hist(spikeIntervals, x)
        # intervalDist2 = intervalDist/sum(intervalDist)

    return spikeTimes