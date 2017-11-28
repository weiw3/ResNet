from adlkit.adlkit.data_provider import H5FileDataProvider

val=[
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]

test=[
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_1.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_2.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_3.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_4.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_5.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_6.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_7.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_8.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_9.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_1.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_2.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_3.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_4.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_5.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_6.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_7.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_8.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_9.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]


train=[
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_10.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_11.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_12.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_13.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_14.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_15.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_16.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_17.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_18.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_19.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_20.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_21.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_22.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_23.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_24.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_25.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_26.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_27.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_28.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/GammaEscan/GammaEscan_29.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_10.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_11.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_12.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_13.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_14.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_15.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_16.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_17.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_18.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_19.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_20.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_21.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_22.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_23.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_24.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_25.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_26.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_27.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_28.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/DownsampledNormalizedGammaPi0MergingSize1Float_COPY/Pi0Escan/Pi0Escan_29.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]

batch_size = 1500

num_gpus = 10

num_gen = 1
readers_per_gen = 10

num_readers=min(num_gen*readers_per_gen, 20)

val_generator = H5FileDataProvider(val, batch_size=batch_size, n_readers=num_readers, q_multipler = 4, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=10000, n_generators=num_gen)
#val_generator = H5FileDataProvider(val, max=20, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, n_generators=2)

test_generator = H5FileDataProvider(test, batch_size=batch_size, n_readers=num_readers, q_multipler = 4, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=100000, n_generators=num_gen)

train_generator = H5FileDataProvider(train, batch_size=batch_size, n_readers=num_readers, q_multipler = 4, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=200000, n_generators=num_gen)
