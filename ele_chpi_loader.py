from adlkit.adlkit.data_provider import H5FileDataProvider

val=[
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_0.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_0.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ]

test=[
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_0.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_1.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_2.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_3.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_4.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_5.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_6.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_7.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_8.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_9.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_0.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_1.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_2.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_3.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_4.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_5.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_6.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_7.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_8.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_9.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ]


train=[
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_10.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_11.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_12.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_13.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_14.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_15.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_16.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_17.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_18.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_19.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_20.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_21.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_22.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_23.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_24.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_25.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_26.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_27.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_28.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/EleEscan/EleEscan_29.h5', ['ECAL', 'HCAL'], 'electron', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_10.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_11.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_12.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_13.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_14.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_15.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_16.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_17.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_18.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_19.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_20.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_21.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_22.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_23.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_24.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_25.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_26.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_27.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_28.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ['/data/LCD/V2/DownsampledNormalizedEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_29.h5', ['ECAL', 'HCAL'], 'chpi', 1],
        ]

batch_size = 1000

num_gpus = 10

num_gen = 16
readers_per_gen = 10

num_readers=min(num_gen*readers_per_gen, 30)

val_generator = H5FileDataProvider(val, batch_size=batch_size, n_readers=num_readers, q_multipler = 4, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=10000, n_generators=num_gen)
#val_generator = H5FileDataProvider(val, max=20, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, n_generators=2)

test_generator = H5FileDataProvider(test, batch_size=batch_size, n_readers=num_readers, q_multipler = 4, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=100000, n_generators=num_gen)

train_generator = H5FileDataProvider(train, batch_size=batch_size, n_readers=num_readers, q_multipler = 4, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=200000, n_generators=num_gen)
