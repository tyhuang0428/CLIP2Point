import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets.modelnet40_align import ModelNet40Ply

from models import DPA
from datasets import ModelNet40Align
from utils import IOStream


clip_model, _ = clip.load('ViT-B/32', device='cpu')
prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe', 'xbox']
prompts = ['image of a ' + prompts[i] for i in range(len(prompts))]
prompts = clip.tokenize(prompts)
prompts = clip_model.encode_text(prompts)
prompts_feats = prompts / prompts.norm(dim=-1, keepdim=True)


def _init_():
    if not os.path.exists('exp_results'):
        os.makedirs('exp_results')
    if not os.path.exists('exp_results/'+args.exp_name):
        os.makedirs('exp_results/'+args.exp_name)


def train(args, io):
    train_dataloader = DataLoader(ModelNet40Align('train', 16), batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(ModelNet40Align('test'), batch_size=args.test_batch_size, num_workers=4, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =================================== INIT MODEL ===========================================================
    model = DPA(args).to(device)
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'selector' not in name and 'renderer' not in name:
            param.requires_grad_(False)
    prompt_feats = prompts_feats.to(device).detach()
    # ==================================== TRAINING LOOP =====================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = args.epoch
    max_test_acc = 0
    summary_writer = SummaryWriter("exp_results/%s/tensorboard" % args.exp_name)
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0
        correct_num = 0
        total = 0
        for (points, label) in tqdm(train_dataloader):
            points = points.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            img_feats = model(points)
            logits = img_feats @ prompt_feats.t()

            loss = F.cross_entropy(logits, label)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(dim=-1)
            index = torch.max(probs, dim=1).indices
            correct_num += torch.sum(torch.eq(index, label)).item()
            total += len(label)
        train_acc = correct_num / total
    
        model.eval()
        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(test_dataloader):
                points = points.to(device)
                img_feats = model(points)
                logits = img_feats @ prompt_feats.t()
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        test_acc = correct_num / total

        mean_loss = loss_sum / len(train_dataloader)
        io.cprint('epoch%d total_loss: %.4f, train_acc: %.4f, test_acc: %.4f' % (epoch + 1, mean_loss, train_acc, test_acc))
        summary_writer.add_scalar('train/loss', mean_loss, epoch + 1)
        summary_writer.add_scalar("train/acc", train_acc, epoch + 1)
        summary_writer.add_scalar("test/acc", test_acc, epoch + 1)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), 'exp_results/%s/best.pth' % (args.exp_name))
            io.cprint('save the best test acc at %d' % (epoch + 1))


def eval(args):
    assert args.ckpt is not None, 'load a checkpoint for evaluation'
    test_dataloader = DataLoader(ModelNet40Ply('test'), batch_size=args.test_batch_size, num_workers=4, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DPA(args, True).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    prompt_feats = prompts_feats.to(device).detach()

    model.eval()
    with torch.no_grad():
        correct_num = 0
        total = 0
        for (points, label) in tqdm(test_dataloader):
            points = points.to(device)
            img_feats = model(points)
            logits = img_feats @ prompt_feats.t()
            probs = logits.softmax(dim=-1)
            index = torch.max(probs, dim=1).indices
            correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
            total += len(label)
    test_acc = correct_num / total
    print(test_acc)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Few-shot Point Cloud Classification')
    parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dim', type=int, default=0, choices=[0, 512], help='0 if the view angle is not learnable')
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['DGCNN', 'PointNet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        _init_()
        io = IOStream('exp_results/' + args.exp_name + '/run.log')
        io.cprint(str(args))
        train(args, io)
    else:
        eval(args)
