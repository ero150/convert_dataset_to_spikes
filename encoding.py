
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from tqdm import tqdm
from snntorch import spikegen

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


def encode_mfcc(inputs, labels, sample_duration, time_step, train_test):
    data = {
        #      "x": [],
        #      "p": [],
        "t": [],
        "label": []
    }

    # output_x = []
    output_t = []
    # output_p = []

    for index_i, i in enumerate(inputs[0]):

        # full_path = os.path.join(NEW_PATH,str(index_i),".json")

        data = {
            "x": [],
            "y": [],
            #      "p": [],
            "t": [],
            "label": []}

        full_path = NEW_PATH + train_test + "scaled10_39mfcc_100ms3d/" + str(index_i) + ".hdf5"
        print(full_path)
        data["label"].append(labels[0][index_i])
        # !!!!!!!!!!

        # add_to_text_file(file1,index_i,labels[0][index_i])

        ###
        counter = 0
        time_counter = -23
        # print("length i:", count(enumerate(i)))

        for index_j, j in enumerate(i):
            # print("length j: ", enumerate(j)[-1])
            # print(data["label"])
            time_counter += 23
            # print(time_counter)
            for index_k, k in enumerate(j):
                # print(k)
                counter += 1
                # print(counter)

                spike_times = encode_spikes(spike_frequency=k, time_duration=100, time_counter=time_counter)

                x = (np.ones(spike_times.shape) * index_k).astype(int)
                y = (np.ones(spike_times.shape) * index_j).astype(int)

                for item in spike_times:
                    data["t"].extend(item)

                # for item in spikes:
                #   data["p"].extend(item)

                for item in x:
                    data["x"].extend(item)

                for item in y:
                    data["y"].extend(item)

            # print("X: ",len(data["x"]))
            # print("Y: ",len(data["y"]))

            # print("index k(mfcc): ",index_k)
            # print("index j(timestep): ",index_j)
            # print("X: ",x)
            # print("Y: ",y)
        # print(data["t"])

        print(len(data["t"]))
        h5file = h5py.File(full_path, "a")  # vytvorenie h5py suboru, mal by mat priponu .hdf5

        h5file.create_dataset(name="label", data=data['label'],
                              compression='gzip')  # pridanie data cez metodu create dataset, bud treba dat na vstup data alebo shape
        h5file.create_dataset(name="x", data=data['x'], compression='gzip')
        h5file.create_dataset(name="y", data=data['y'], compression='gzip')
        h5file.create_dataset(name="t", data=data['t'], compression='gzip')
        h5file.close()


def encode_mfcc2(inputs, labels, sample_duration, num_steps, train_test,new_path):
    filename_id = 0
    for a in tqdm(range(len(inputs))):
        # print("file: ",filename_id)

        data_it = np.array(inputs[a])
        label = labels[a]
        # print("label: ", label)
        final_x = []
        final_y = []
        final_t = []
        label_array = []
        # Spiking Data
        # spike_data = spikegen.rate(data_it, num_steps=num_steps)

        t = torch.from_numpy(data_it)
        #spike_data = spikegen.latency(t, num_steps=num_steps, tau=5, threshold=0.01)
        spike_data = spikegen.rate(t, num_steps=num_steps)
        spike_data = spike_data.cpu().detach().numpy()
        a = 0
        # print(spike_data.shape)
        for index_timestep, timestep in enumerate(spike_data):
            # print(index_timestep,'timestep',timestep.shape)
            # print(index_batch,'batch',batch.shape)
            # print(index_dim,'dim',dim.shape)
            # counter+=1
            for index_x, x in enumerate(timestep):
                # print(index_x,'x',x.shape)
                # for y in x:
                temp_y_indexes = []
                y_indexes = np.where(x > 0)
                # print(y_indexes)

                for y_index in y_indexes:
                    temp_y_indexes.extend(y_index)

                    temp_x = np.ones(len(temp_y_indexes), dtype=int) * index_x
                    temp_t = np.ones(len(temp_y_indexes), dtype=int) * index_timestep

                    final_x.extend(temp_x)
                    final_y.extend(temp_y_indexes)
                    final_t.extend(temp_t)

        # break
        # print(spike_data.shape)
        # print("x",len(final_x))
        # print(final_x)
        # print("y",len(final_y))
        # print(final_y)
        # print("t",len(final_t))
        # print(final_t)

        full_path = new_path +train_test+'_'+str(num_steps)+'ms/'+str(filename_id)+".hdf5"
        h5file = h5py.File(full_path,"a") # vytvorenie h5py suboru, mal by mat priponu .hdf5
        label_array.append(label)
        h5file.create_dataset(name="label",data = label_array,compression='gzip') # pridanie data cez metodu create dataset, bud treba dat na vstup data alebo shape
        h5file.create_dataset(name="x",data = final_x,compression='gzip')
        h5file.create_dataset(name="y",data = final_y,compression='gzip')
        h5file.create_dataset(name="t",data = final_t,compression='gzip')
        h5file.close()

        filename_id += 1