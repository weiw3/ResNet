from adlkit.adlkit.data_provider import H5FileDataProvider

signal_test=[
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_1.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_2.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_3.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_4.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_5.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_6.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_7.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_8.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_9.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ]

background_test=[
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_1.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_2.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_3.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_4.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_5.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_6.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_7.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_8.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_9.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]

val=[
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]

test=[
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_1.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_2.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_3.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_4.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_5.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_6.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_7.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_8.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTest/Gamma_9.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_1.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_2.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_3.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_4.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_5.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_6.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_7.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_8.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Test/Pi0_9.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]


signal_train=[
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_1.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_2.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_3.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_4.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_5.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_6.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_7.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_8.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_9.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_10.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_11.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_12.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_13.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_14.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_15.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_16.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_17.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_18.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_19.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ]

background_train=[
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_1.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_2.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_3.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_4.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_5.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_6.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_7.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_8.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_9.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_10.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_11.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_12.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_13.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_14.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_15.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_16.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_17.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_18.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_19.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]
train=[
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_0.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_1.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_2.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_3.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_4.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_5.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_6.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_7.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_8.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_9.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_10.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_11.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_12.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_13.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_14.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_15.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_16.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_17.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_18.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/GammaTrain/Gamma_19.h5', ['ECAL', 'HCAL'], 'gamma', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_0.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_1.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_2.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_3.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_4.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_5.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_6.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_7.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_8.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_9.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_10.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_11.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_12.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_13.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_14.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_15.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_16.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_17.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_18.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ['/data/LCD/V2/SimpleGammaPi0/Pi0Train/Pi0_19.h5', ['ECAL', 'HCAL'], 'pi0', 1],
        ]

batch_size = 1000

val_generator = H5FileDataProvider(val, batch_size=batch_size, n_readers=10, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=10000)
#val_generator = H5FileDataProvider(val, max=20, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, n_generators=2)

test_generator = H5FileDataProvider(test, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=False, max_samples=100000)

train_generator = H5FileDataProvider(train, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=True, max_samples=200000)

signal_test_generator = H5FileDataProvider(signal_test, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=False, max_samples=100000)

background_test_generator = H5FileDataProvider(background_test, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=False, max_samples=100000)

signal_train_generator = H5FileDataProvider(signal_train, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=False, max_samples=200000)

background_train_generator = H5FileDataProvider(background_train, batch_size=batch_size, n_readers=30, q_multipler = 2, read_multiplier = 1, make_class_index=True, sleep_duration=1, wrap_examples=False, max_samples=200000)
