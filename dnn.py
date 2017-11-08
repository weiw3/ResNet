from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import numpy as np
import h5py

import os
from torch import cat
from torch import from_numpy

from multiprocessing import Process, Pool

execfile("ele_chpi_loader.py")

OutPath='test/'

def objective(params, GENERATOR_ID):
# define the model
    print params
    #print objective.index_of_call
    #objective.index_of_call+=1
    depth, width = params
    learning_rate=0.001
    decay_rate=0
    class Net2(nn.Module):
        def __init__(self):
            super(Net2, self).__init__()
            self.fc1 = nn.Linear(25 * 25 * 25 + 5 * 5 * 60, width)
            self.fc5 = nn.Linear(width, width)
            self.fc4 = nn.Linear(width, 2)
            self.norm = nn.BatchNorm1d(width)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x1, x2):
            x1 = x1.view(-1, 25 * 25 * 25)
            x2 = x2.view(-1, 5 * 5 * 60)
        
            x = cat((x1,x2), 1)

            x = self.fc1(x)
            x = self.norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            for _ in range(depth-1):
    #            x = self.fc5(x)
    #            x = self.norm(x)
    #            x = F.relu(x)
    #            x = self.dropout(x)
                y = self.fc5(x)
                y = self.norm(y)
                y = F.relu(y)
                y = self.fc5(y)
                y = self.norm(y)
                y = F.relu(y)
                #y = self.dropout(y)
                x = F.relu(self.norm(self.norm(self.fc5(y))+x))
                x = self.dropout(x)

            #x = self.dropout(x)
            x = self.fc4(x)

            return x


    from torch import load

    #net = nn.DataParallel(Net2())
    net = Net2()

# load previous model
    #net.load_state_dict(load("../Downsampled_GammaPi0_1_merged_nn_outputs/savedmodel_32-6_lr-0.0002_dr-0_10-15"))

    net.cuda()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)


    loss_history = []
    epoch_end_relative_error_history = []


    epoch_num=50


# main process for training
    prev_val_loss = 0.0
    stag_break_count = 0
    over_break_count = 0
    prev_epoch_end_val_loss = 0.0
    epoch_end_val_loss = 0.0

    #train_generator.start()
    train_loader = train_generator.generators[GENERATOR_ID].generate()
    
    #val_generator.start()
    val_loader = val_generator.generators[GENERATOR_ID].generate()
    
    running_loss = 0.0
    val_loss = 0.0
    i = 0
    import itertools
    for train_data, val_data in itertools.izip(train_loader, val_loader):
        #inputs, labels = data
        #ECAL, HCAL, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda()), Variable(labels.cuda())
        ECAL, HCAL, labels = train_data
        ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), Variable(from_numpy(labels).long().cuda())
        optimizer.zero_grad()
        outputs = net(ECAL, HCAL)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]



        ECAL, HCAL, labels = val_data
        ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), Variable(from_numpy(labels).long().cuda())
        val_outputs = net(ECAL, HCAL)
        validation_loss = criterion(val_outputs, labels)
        val_loss += validation_loss.data[0]

        if i % 20 == 19:
            running_loss /= 20
            val_loss /= 20
            print('[%d, %5d, %5d] loss: %.10f' %
                    (GENERATOR_ID, i/400 + 1, i%400 + 1, running_loss)),
            print('    val loss: %.10f' %
                    (val_loss)),
            relative_error = (val_loss-prev_val_loss)/float(val_loss)
            print('    relative error: %.10f' %
                    (relative_error)),

            if(relative_error>0.03 and i!=0):
                over_break_count+=1
                if(over_break_count>2):
                    break
            else:
                over_break_count=0
            
            print('    over break count: %d' %
                    (over_break_count))
            
            loss_history.append([GENERATOR_ID, i/400 + 1, i%400 + 1, running_loss, val_loss, relative_error, over_break_count])
            
            if(i % 400==399):
                epoch_end_val_loss = val_loss
                epoch_end_relative_error = (epoch_end_val_loss-prev_epoch_end_val_loss)/float(epoch_end_val_loss)
                print('[%d] epoch_end_relative_error: %.10f' %
                        (GENERATOR_ID, epoch_end_relative_error)),
                epoch_end_relative_error_history.append([GENERATOR_ID, i/400 + 1, i%400 + 1, epoch_end_relative_error])

                if(epoch_end_relative_error > -0.005 and i/400!=0):
                    stag_break_count+=1
                    if(stag_break_count>0):
                        break
                else:
                    stag_break_count=0
                print('    stag_break_count: %d' %
                        (stag_break_count))
                prev_epoch_end_val_loss = epoch_end_val_loss

            
            prev_val_loss = val_loss
            running_loss = 0.0
            val_loss = 0.0
        i+=1

    #train_generator.hard_stop()
    #val_generator.hard_stop()
        #break;

    loss_history = np.array(loss_history)
    epoch_end_relative_error_history = np.array(epoch_end_relative_error_history)
    with h5py.File(OutPath+"loss_history-depth_"+str(depth)+"-width_"+str(width)+".h5", 'w') as loss_file, h5py.File(OutPath+"epoch_end_relative_error_history-depth_"+str(depth)+"-width_"+str(width)+".h5", 'w') as epoch_end_relative_error_history_file:
        loss_file.create_dataset("loss", data=loss_history)
        epoch_end_relative_error_history_file.create_dataset("relative_error", data=epoch_end_relative_error_history)

    from torch import save
    save(net.state_dict(), OutPath+"savedmodel_depth_"+str(depth)+"-width_"+str(width))

    print('Finished Training')

# Analysis
    from torch import max

    correct = 0
    total = 0
    #test_generator.start()
    test_loader = test_generator.generators[GENERATOR_ID].generate()
    test_count = 0
    for test_index, test_data in enumerate(test_loader):
        test_count += 1
        #images, labels = data
        #ECAL, HCAL, labels = Variable(images[0].cuda()), Variable(images[1].cuda()), labels.cuda()
        #outputs = net(ECAL, HCAL)
        ECAL, HCAL, labels = test_data
        ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), from_numpy(labels).long().cuda()
        outputs = net(ECAL, HCAL)
        _, predicted = max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        if(test_count >= 200):
            break;
    #test_generator.hard_stop()
    print('Accuracy of the network on test images: %f %%' % (
            100 * float(correct) / total))



    #from torch import Tensor

    #outputs0=Tensor().cuda()
    #outputs1=Tensor().cuda()
    #outputs2=Tensor().cuda()
    #outputs3=Tensor().cuda()


#  separate outputs for training/testing signal/backroung events. 
    #signal_train_generator.start()
    #signal_train_generator_count = 0 
    #signal_train_loader = signal_train_generator.generators[GENERATOR_ID].generate()
    #for data in signal_train_loader:
        #signal_train_generator_count += 1
        #ECAL, HCAL, labels = data
        #ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), from_numpy(labels).long().cuda()
        #outputs0 = cat((outputs0, net(ECAL, HCAL).data))

        #if signal_train_generator_count >= 200:
            #break;
    #signal_train_generator.hard_stop()

    #signal_test_generator.start()
    #signal_test_generator_count = 0 
    #signal_test_loader = signal_test_generator.generators[GENERATOR_ID].generate()
    #for data in signal_test_loader:
        #signal_test_generator_count += 1
        #ECAL, HCAL, labels = data
        #ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), from_numpy(labels).long().cuda()
        #outputs1 = cat((outputs1, net(ECAL, HCAL).data))

        #if signal_test_generator_count >= 100:
            #break;
    #signal_test_generator.hard_stop()


    #background_train_generator.start()
    #background_train_generator_count = 0 
    #background_train_loader = background_train_generator.generators[GENERATOR_ID].generate()
    #for data in background_train_loader:
        #background_train_generator_count += 1
        #ECAL, HCAL, labels = data
        #ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), from_numpy(labels).long().cuda()
        #outputs2 = cat((outputs2, net(ECAL, HCAL).data))
        #if background_train_generator_count >= 200:
            #break;
    #background_train_generator.hard_stop()


    #background_test_generator.start()
    #background_test_generator_count = 0 
    #background_test_loader = background_test_generator.generators[GENERATOR_ID].generate()
    #for data in background_test_loader:
        #background_test_generator_count += 1
        #ECAL, HCAL, labels = data
        #ECAL, HCAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(HCAL).cuda()), from_numpy(labels).long().cuda()
        #outputs3 = cat((outputs3, net(ECAL, HCAL).data))

        #if background_test_generator_count >= 100:
            #break;
    #background_test_generator.hard_stop()


    #with h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_0.h5", 'w') as o1, h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_1.h5", 'w') as o2, h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_2.h5", 'w') as o3, h5py.File(OutPath+"out_depth_"+str(depth)+"-width_"+str(width)+"_3.h5", 'w') as o4:
        #o1.create_dataset("output", data=outputs0.cpu().numpy())
        #o2.create_dataset("output", data=outputs1.cpu().numpy())
        #o3.create_dataset("output", data=outputs2.cpu().numpy())
        #o4.create_dataset("output", data=outputs3.cpu().numpy())

    return (float(correct) / total)*100.0


def run(pid):
    width_min=256
    width_max=511
    depth = 30 
    depth_min = 5 
    depth_max = 5 

    accuracies = []

    with torch.cuda.device(pid%num_gpus):
        for width in range(width_min + pid, width_max + 1, num_gen):
        #import rpdb; rpdb.set_trace()
            accuracy = objective((depth, width), pid)

            accuracies.append([width, accuracy])

    with h5py.File(OutPath+"accuracies_depth_range_"+str(depth_min)+"-"+str(depth_max)+"_width_range_"+str(width_min)+"-"+str(width_max)+"_"+str(pid)+".h5", 'w') as write_accuracies:
        write_accuracies.create_dataset('accuracies', data=np.array(accuracies))


if __name__ == '__main__':

    num_of_processes = num_gen


    #objective.index_of_call=0


    train_generator.start()
    val_generator.start()
    test_generator.start()

    worker_pool = Pool(processes=num_of_processes)

    worker_pool.map(run, range(num_of_processes))


    train_generator.hard_stop()
    val_generator.hard_stop()
    test_generator.hard_stop()
