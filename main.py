"""
Entry point for training and testing IMU transformers for activity recognition
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
from os.path import join
from models.IMUTransformerEncoder import IMUTransformerEncoder
from models.IMUCLSBaseline import IMUCLSBaseline
from util.IMUDataset import IMUDataset
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or test")
    arg_parser.add_argument("imu_dataset_file", help="path to a file mapping imu samples to labels")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {}ing IMU-transformers".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using imu dataset file: {}".format(args.imu_dataset_file))

    # Read configuration
    with open('config.json', "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    if config.get("use_baseline"):
        model = IMUCLSBaseline(config).to(device)
    else:
        model = IMUTransformerEncoder(config).to(device)

    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        loss = torch.nn.NLLLoss()

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        logging.info("Start train data preparation")
        window_shift = config.get("window_shift")
        window_size = config.get("window_size")
        input_size = config.get("input_dim")

        dataset = IMUDataset(args.imu_dataset_file, window_size, input_size, window_shift)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        logging.info("Data preparation completed")

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        logging.info("Start training")
        for epoch in range(n_epochs):

            for batch_idx, minibatch in enumerate(dataloader):
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long)
                batch_size = label.shape[0]

                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Collect for recoding and plotting
                batch_loss = criterion.item()
                loss_vals.append(batch_loss)
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("[Batch-{}/Epoch-{}] batch loss: {:.3f}".format(
                                                                        batch_idx+1, epoch+1,
                                                                        batch_loss))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # Plot the loss function
        #loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        #utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        logging.info("Start test data preparation")
        window_shift = config.get("window_shift")
        window_size = config.get("window_size")
        input_size = config.get("input_dim")
        dataset = IMUDataset(args.imu_dataset_file, window_size, input_size,
                             window_shift)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        logging.info("Data preparation completed")

        metric = []
        logging.info("Start testing")
        accuracy_per_label = np.zeros(config.get("num_classes"))
        count_per_label = np.zeros(config.get("num_classes"))
        predicted = []
        ground_truth = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):

                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.long)

                # Forward pass
                res = model(minibatch)

                # Evaluate and append
                pred_label = torch.argmax(res)
                predicted.append(pred_label.cpu().numpy())
                ground_truth.append( label[0].item())
                curr_metric = (pred_label==label).to(torch.int)
                label_id = label[0].item()
                accuracy_per_label[label_id] += curr_metric.item()
                count_per_label[label_id] += 1
                metric.append(curr_metric.item())

        # Record overall statistics
        stats_msg = "Performance of {} on {}".format(args.checkpoint_path, args.imu_dataset_file)
        confusion_mat = confusion_matrix(ground_truth, predicted, labels = list(range(config.get("num_classes"))))
        print(confusion_mat.shape)
        stats_msg = stats_msg + "\n\tAccuracy: {:.3f}".format(np.mean(metric))
        accuracies = []
        for i in range(len(accuracy_per_label)):
                print("Performance for class [{}] - accuracy {:.3f}".format(i, accuracy_per_label[i]/count_per_label[i]))
                accuracies.append(accuracy_per_label[i]/count_per_label[i])
        # save dump
        np.savez(args.checkpoint_path + "_test_results_dump", confusion_mat = confusion_mat, accuracies = accuracies, count_per_label=count_per_label, total_acc = np.mean(metric))
        logging.info(stats_msg)

