import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm
import clip
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter

from models import CLIP2Point
from datasets import ModelNet40Align, ShapeNetRender
from utils import IOStream

clip_model, _ = clip.load("ViT-B/32", device='cpu')


def _init_(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/' + args.exp_name):
        os.makedirs(path + '/' + args.exp_name)


def train(args, io):
    test_prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe', 'xbox']
    val_prompts = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer']
    test_prompts = ['image of a ' + test_prompts[i] for i in range(len(test_prompts))]
    val_prompts = ['image of a ' + val_prompts[i] for i in range(len(val_prompts))]
    test_prompts_ = clip.tokenize(test_prompts)
    test_prompt_feats = clip_model.encode_text(test_prompts_)
    test_prompt_feats = test_prompt_feats / test_prompt_feats.norm(dim=-1, keepdim=True)
    test_prompt_feats = test_prompt_feats
    val_prompts_ = clip.tokenize(val_prompts)
    val_prompt_feats = clip_model.encode_text(val_prompts_)
    val_prompt_feats = val_prompt_feats / val_prompt_feats.norm(dim=-1, keepdim=True)
    val_prompt_feats = val_prompt_feats

    train_dataloader = DataLoader(ShapeNetRender(partition='train', num_points=args.num_points), batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(ShapeNetRender(partition='test', num_points=args.num_points), batch_size=args.test_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(ModelNet40Align(num_points=args.num_points), batch_size=args.test_batch_size, num_workers=4)
    gpu_num = torch.cuda.device_count()
    gpus = [i for i in range(gpu_num)]
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    # =================================== INIT MODEL ==========================================================
    summary_writer = SummaryWriter("pre_results/%s/tensorboard" % (args.exp_name))
    model = CLIP2Point(args)
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])  # 多卡训练修改
    model = model.to(device)
    for name, param in model.named_parameters():
        if 'image_model' in name:
            param.requires_grad_(False)
    val_prompt_feats = val_prompt_feats.to(device)
    test_prompt_feats = test_prompt_feats.to(device)
    # ==================================== TRAINING LOOP ======================================================
    optimizer = optim.Lamb(model.parameters(), lr=0.006, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(train_dataloader),
        T_mult=1,
        eta_min=max(1e-2 * 1e-3, 1e-6),
        last_epoch=-1,
    )

    n_epochs = args.epoch
    max_val_acc = 0
    max_test_acc = 0
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0
        depth_sum = 0
        image_sum = 0
        
        for (image, points, a, e, d) in tqdm(train_dataloader):
            optimizer.zero_grad()
            image = image.to(device)
            points = points.to(device)
            a = a.unsqueeze(-1).to(device)
            e = e.unsqueeze(-1).to(device)
            d = d.unsqueeze(-1).to(device)
            loss, image_loss, depth_loss = model(points, image, a, e, d)
            loss = torch.mean(loss)
            image_sum += torch.mean(image_loss).item()
            depth_sum += torch.mean(depth_loss).item()
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
        # Validation and Testing
        model.eval()
        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(val_loader):
                b = points.shape[0]
                points = points.to(device)
                img_feats = model.module.infer(points)

                logits = img_feats @ val_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        val_acc = correct_num / total

        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(test_loader):
                b = points.shape[0]
                points = points.to(device)
                img_feats = model.module.infer(points, True)
                logits = img_feats @ test_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        test_acc = correct_num / total

        depth_loss = depth_sum / len(train_dataloader)
        image_loss = image_sum / len(train_dataloader)
        mean_loss = loss_sum / len(train_dataloader)
        io.cprint('epoch%d total_loss: %.4f, image_loss: %.4f, depth_loss: %.4f, balance_weights: %.4f, val_acc: %.4f, test_acc: %.4f' % (epoch + 1, mean_loss, image_loss, depth_loss, model.module.weights, val_acc, test_acc))
        summary_writer.add_scalar('train/loss', mean_loss, epoch + 1)
        summary_writer.add_scalar('train/depth_loss', depth_loss, epoch + 1)
        summary_writer.add_scalar('train/image_loss', image_loss, epoch + 1)
        summary_writer.add_scalar('train/weights', model.module.weights, epoch + 1)
        summary_writer.add_scalar("val/acc", val_acc, epoch + 1)
        summary_writer.add_scalar("test/acc", test_acc, epoch + 1)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), '%s/%s/best_val.pth' % ('pre_results', args.exp_name))
            io.cprint('save the best val acc at %d' % (epoch + 1))
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), '%s/%s/best_test.pth' % ('pre_results', args.exp_name))
            io.cprint('save the best test acc at %d' % (epoch + 1))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dim', type=int, default=0, choices=[0, 512], help='0 if the view angle is not learnable')
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['DGCNN', 'PointNet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()

    _init_('pre_results')
    io = IOStream('pre_results' + '/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    train(args, io)
