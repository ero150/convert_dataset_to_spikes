def scale_inputs(inputs, absolute_max, absolute_min, new_max=1, new_min=0):
    # absolute_max = np.max(inputs)
    # absolute_min = np.min(inputs)

    scaled = inputs
    # print(scaled[0][0])

    for index_i, i in enumerate(scaled):
        # print("Scaled length", len(scaled))
        # print("I length", len(i))
        for index_j, j in enumerate(i):
            # print(j)
            # print("J lenght",len(j))
            # print("Index ", index_j)
            # print(j)
            # X_std = (j - absolute_min) / (absolute_max - absolute_min)
            # scaled[index_i][index_j] = X_std * (new_max - new_min) + new_min
            scaled[index_i][index_j] = (j - absolute_min) / (absolute_max - absolute_min)

    # print(scaled[0][0])
    return scaled