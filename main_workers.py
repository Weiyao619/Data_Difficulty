import models
import warnings
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.optim import lr_scheduler
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset
from utils import *
from train_valid import *
from parser_args import *

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = 'cuda:0'
    Data = args.dataset

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    scheduler = lr_scheduler.MultiStepLR(optimizer, [30,100], gamma = 0.1)

    #cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    if Data == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_valid)
    elif Data == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_valid)
    elif Data is not None:
        train_dataset = torch.load('trainset_'+Data+'.pt')
        if args.cifar_10or100 == '10':
            val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_valid)
        else :
            val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_valid)
    else:
        warnings.warn('Dataset is not listed')
        return
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)

    validloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train_pbar.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test_pbar.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    #pth = '/root/Stage/Experiments/CleanPart/checkpoint/cifar10_ResNet34_CE_None_exp_0.01_0/ckpt.best.pth.tar'
    #if pth is not None:
        #checkpoint = torch.load(pth)
        #model.load_state_dict(checkpoint['state_dict'])
    for epoch in range(args.start_epoch, args.epochs):

        
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # train for one epoch
        if args.clean_noisy == 'clean':
            train(trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        else:
            trainNoisy(trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1 = validate(validloader, model, criterion, epoch, args, log_testing, tf_writer)
        print("epoch:",epoch,"current accuracy:",acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print("epoch:",epoch,output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        scheduler.step()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=args.momentum, weight_decay=args.weight_decay) 
    scheduler = lr_scheduler.MultiStepLR(optimizer, [1,20,25], gamma = 0.1)
    checkpoint = torch.load('/root/Stage/Experiments/CleanPart/checkpoint/'+args.dataset+'_ResNet34_CE_None_exp_0.01_0/ckpt.best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    for epoch in range(args.start_epoch, args.Re_epochs):
    
        
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # train for one epoch
        if args.clean_noisy == 'clean':
            train(trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        else:
            trainNoisy(trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1 = validate(validloader, model, criterion, epoch, args, log_testing, tf_writer)
        print("epoch:",epoch,"current accuracy:",acc1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print("epoch:",epoch,output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        scheduler.step()

def main_worker_pbar(gpu, args):
    global best_acc1
    p = torch.empty((args.K,args.N,50000,10))
    softmax = nn.Softmax(dim = 1)
    args.gpu = 'cuda:0'

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_valid)
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_valid)
    elif args.dataset is not None:
        train_dataset = torch.load('trainset_'+args.dataset+'.pt')
        if args.cifar_10or100 == '10':
            val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_valid)
        else :
            val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_valid)
    else:
        warnings.warn('Dataset is not listed')
        return
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)

    validloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train_pbar.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test_pbar.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for k in range(args.K):
        for n in range(args.N):
            split1 = n/args.N
            split2 = (n+1)/args.N
            dataset_size = len(train_dataset)
            #print(dataset_size)
            indices = list(range(dataset_size))
            split_place1 = int(np.floor(split1 * dataset_size))
            split_place2 = int(np.floor(split2 * dataset_size))-1
            #print("splites:",split1,split2)
            print("split_places:",split_place1,split_place2)
            mini_train_indices = indices[:split_place1]+indices[split_place2:]
            print(len(mini_train_indices))
            mini_train_sampler = SubsetRandomSampler(mini_train_indices)
            mini_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,sampler=mini_train_sampler)
            
            best_acc1 = 0
            optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
            scheduler = lr_scheduler.MultiStepLR(optimizer, [30,100], gamma = 0.1)
            checkpoint = torch.load('/root/Stage/Experiments/CleanPart/checkpoint/cifar10_ResNet34_CE_None_exp_0.01_0/ckpt.best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            print("model loaded with best accuracy:",best_acc1,"noisy or clean:",args.clean_noisy)
            for epoch in range(args.start_epoch, 150):
                criterion = nn.CrossEntropyLoss().cuda(args.gpu)


                # train for one epoch
                if args.clean_noisy == 'clean':
                    train(mini_trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
                else:
                    trainNoisy(mini_trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
                # evaluate on validation set
                acc1 = validate(validloader, model, criterion, epoch, args, log_testing, tf_writer)
                #print("current accuracy:",acc1)

            # remember best acc@1 and save checkpoint
                if  epoch % 10 == 9 :
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)

                    tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
                    output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
                    output_acc1 = 'Prec@1: %.3f' % acc1
                    if epoch%10==9:
                        print("epoch:",epoch+1,output_acc1,output_best)
                    log_testing.write(output_best + '\n')
                    log_testing.flush()

                    save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)

                scheduler.step()

            optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=args.momentum, weight_decay=args.weight_decay) 
            scheduler = lr_scheduler.MultiStepLR(optimizer, [1,20,25], gamma = 0.1)
            checkpoint = torch.load('/root/Stage/Experiments/CleanPart/checkpoint/'+args.dataset+'_ResNet34_CE_None_exp_0.01_0/ckpt.best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            for epoch in range(args.start_epoch, 30):
                criterion = nn.CrossEntropyLoss().cuda(args.gpu)


                # train for one epoch
                if args.clean_noisy == 'clean':
                    train(mini_trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
                else:
                    trainNoisy(mini_trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
                # evaluate on validation set
                acc1 = validate(validloader, model, criterion, epoch, args, log_testing, tf_writer)
                #print("current accuracy:",acc1)

            # remember best acc@1 and save checkpoint
                if epoch >=5 :
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)

                    tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
                    output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
                    output_acc1 = 'Prec@1: %.3f' % acc1
                    if epoch%10==9:
                        print("epoch:",epoch+1,output_acc1,output_best)
                    log_testing.write(output_best + '\n')
                    log_testing.flush()

                    save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)

                scheduler.step()

            checkpoint = torch.load('/root/Stage/Experiments/CleanPart/checkpoint/'+args.dataset+'_ResNet34_CE_None_exp_0.01_0/ckpt.best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            with torch.no_grad():
                for i,data in enumerate(trainloader, 0):
                    if args.clean_noisy == 'clean':
                        input, target = data
                    else:
                        input, target,change = data
                    if args.gpu is not None:
                        input = input.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                    # compute output
                    output = model(input)
                    Out_soft = softmax(output)
                    for j in range(target.size(0)):
                        p[k,n,128*i+j] = Out_soft[j]
            print("k:",k+1,"n:",n+1)
                
    pth = '/root/Stage/Experiments/CleanPart/ptFile/' + args.dataset
    torch.save(p,pth+'/P.pt')
    Pbar = trainpbar(p,args.K,args.N)
    torch.save(Pbar,pth+'/Pbar.pt')
    Var,Bias,Error= trainErr(trainloader,p,Pbar,args)
    torch.save(Var,pth+'/Variance.pt')
    torch.save(Bias,pth+'/Bias.pt')
    torch.save(Error,pth+'/Error.pt')                       
            
def main_worker_l(gpu, args):
    global best_acc1
    L = np.zeros(50000)
    softmax = nn.Softmax(dim = 1)
    args.gpu = 'cuda:0'
    Data = args.dataset

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20,40,60,160], gamma = 0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    if Data == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_valid)
    elif Data == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_valid)
    elif Data is not None:
        train_dataset = torch.load('trainset_'+Data+'.pt')
        if args.cifar_10or100 == '10':
            val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_valid)
        else :
            val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_valid)
    else:
        warnings.warn('Dataset is not listed')
        return
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)

    validloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train_l.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test_l.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    checkpoint = torch.load('checkpoint/cifar10_resnet32_CE_None_exp_0.01_0/ckpt.best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for epoch in range(args.start_epoch, args.epochs):

        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        else:
            warnings.warn('Sample rule is not listed')
        
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        if args.clean_noisy == 'clean':
            train(trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        else:
            trainNoisy(trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
        # evaluate on validation set
        acc1 = validate(validloader, model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        if epoch%10==9:
            print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        scheduler.step()

        if epoch %20 == 19:
            k = (epoch+1)%20-1
            with torch.no_grad():
                for i,data in enumerate(trainloader, 0):
                    input, target,change = data
                    if args.gpu is not None:
                        input = input.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                    # compute output
                    output = model(input)
                    Out_soften = softmax(output)
                    for j in range(target.size(0)):
                        y = Out_soften[j].cpu().numpy().reshape(10)
                        t = target[j].cpu().numpy().reshape(1)
                        L[128*i+j] += cross_entropy_error(y,t)

    L = L/20
    pth = '/root/Stage/Experiments/CleanPart/ptFile/' + Data
    torch.save(L,pth+'/L.pt')