"""
Learning Rate Finder Script for Error Recognition Model
Performs a learning rate range test to find optimal learning rate
"""

import sys
import os
import argparse

# Parse arguments FIRST to determine device BEFORE importing torch
parser = argparse.ArgumentParser(description='Learning Rate Finder')
parser.add_argument('--method', type=str, default='finder', 
                   choices=['finder', 'grid'],
                   help='Method to use: "finder" for fast LR range test, "grid" for grid search')
parser.add_argument('--device', type=str, default='cuda',
                   help='Device to use (cuda or cpu)')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                   help='Weight decay for optimizer')
parser.add_argument('--variant', type=str, default="MLP", help='variant')

args, unknown = parser.parse_known_args()

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch
if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# NOW import torch and all other modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from base import fetch_model, train_step_test_step_dataset_base, test_er_model, fetch_model_name, CUSTOM_THRESHOLD
from core.config import Config
from constants import Constants as const


class LearningRateFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"lr": [], "loss": []}
        
    def range_test(self, train_loader, val_loader, start_lr=1e-7, end_lr=1, num_iter=100, 
                   smooth_f=0.05, diverge_th=5):
        """
        Perform learning rate range test.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Threshold for stopping if loss diverges
        """
        # Save initial model state
        initial_state = deepcopy(self.model.state_dict())
        
        # Set model to training mode
        self.model.train()
        
        # Calculate learning rate multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        best_loss = float('inf')
        batch_num = 0
        losses = []
        log_lrs = []
        
        # Create iterator
        iterator = iter(train_loader)
        
        pbar = tqdm(range(num_iter), desc="LR Finder")
        
        for iteration in pbar:
            batch_num += 1
            
            # Get next batch (cycle through dataloader if needed)
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data, target = next(iterator)
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\nNaN loss at lr={lr:.2e}, stopping...")
                break
            
            # Compute smoothed loss
            if iteration == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            
            # Record best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Check if loss has diverged
            if avg_loss > diverge_th * best_loss:
                print(f"\nLoss diverged at lr={lr:.2e}, stopping...")
                break
            
            # Store values
            self.history["lr"].append(lr)
            self.history["loss"].append(avg_loss)
            losses.append(avg_loss)
            log_lrs.append(np.log10(lr))
            
            # Update progress bar
            pbar.set_postfix({"lr": f"{lr:.2e}", "loss": f"{avg_loss:.4f}"})
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        # Restore initial model state
        self.model.load_state_dict(initial_state)
        
        return self.history
    
    def plot(self, skip_start=10, skip_end=5, suggest=True, log_scale=True):
        """
        Plot learning rate vs loss.
        
        Args:
            skip_start: Number of batches to skip at the start
            skip_end: Number of batches to skip at the end
            suggest: Whether to suggest learning rate
            log_scale: Whether to use log scale for x-axis
        """
        if not self.history["lr"]:
            print("No data to plot. Run range_test() first.")
            return
        
        lrs = self.history["lr"][skip_start:-skip_end] if skip_end > 0 else self.history["lr"][skip_start:]
        losses = self.history["loss"][skip_start:-skip_end] if skip_end > 0 else self.history["loss"][skip_start:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if log_scale:
            ax.semilogx(lrs, losses)
        else:
            ax.plot(lrs, losses)
        
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Finder")
        ax.grid(True)
        
        if suggest:
            # Suggest learning rate based on steepest descent
            min_grad_idx = None
            min_grad = float('inf')
            
            for i in range(1, len(losses) - 1):
                grad = (losses[i+1] - losses[i-1]) / 2
                if grad < min_grad:
                    min_grad = grad
                    min_grad_idx = i
            
            if min_grad_idx is not None:
                suggested_lr = lrs[min_grad_idx]
                ax.scatter(suggested_lr, losses[min_grad_idx], color='red', s=100, 
                          zorder=5, label=f'Suggested LR: {suggested_lr:.2e}')
                ax.legend()
                print(f"\nSuggested learning rate: {suggested_lr:.2e}")
                print(f"You can also try 1/10 of this value: {suggested_lr/10:.2e}")
        
        plt.tight_layout()
        plt.savefig('lr_finder_plot.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'lr_finder_plot.png'")
        plt.show()
        
        return fig


def grid_search_lr(config, lr_values, num_epochs=5):
    """
    Perform grid search over different learning rates.
    
    Args:
        config: Configuration object
        lr_values: List of learning rates to test
        num_epochs: Number of epochs to train for each lr
    """
    results = []
    
    for lr in lr_values:
        print(f"\n{'='*60}")
        print(f"Testing Learning Rate: {lr:.2e}")
        print(f"{'='*60}\n")
        
        # Create fresh model
        config.lr = lr
        config.num_epochs = num_epochs
        
        # Get data loaders
        train_loader, val_loader, test_loader = train_step_test_step_dataset_base(config)
        
        # Create model and optimizer
        model = fetch_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        best_val_auc = 0
        
        for epoch in range(1, num_epochs + 1):
            # Training
            model.train()
            train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
            for data, target in train_pbar:
                data, target = data.to(config.device), target.to(config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                if torch.isnan(loss):
                    print(f"NaN loss encountered at lr={lr:.2e}, skipping...")
                    break
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if len(train_losses) == 0:
                print(f"Training failed for lr={lr:.2e}")
                break
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation - pass CUSTOM_THRESHOLD to test_er_model
            val_losses, sub_step_metrics, step_metrics = test_er_model(
                model, val_loader, criterion, config.device, phase='val',
                threshold=CUSTOM_THRESHOLD
            )
            avg_val_loss = np.mean(val_losses)
            val_auc = step_metrics['auc']
            
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val AUC={val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        results.append({
            'lr': lr,
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc
        })
        
        print(f"\nLR {lr:.2e}: Best Val Loss={best_val_loss:.4f}, Best Val AUC={best_val_auc:.4f}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("LEARNING RATE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Using threshold: {CUSTOM_THRESHOLD}")
    print(f"{'LR':<15} {'Best Val Loss':<20} {'Best Val AUC':<15}")
    print(f"{'-'*60}")
    
    for result in results:
        print(f"{result['lr']:<15.2e} {result['best_val_loss']:<20.4f} {result['best_val_auc']:<15.4f}")
    
    # Find best learning rate based on validation AUC
    best_result = max(results, key=lambda x: x['best_val_auc'])
    print(f"\n{'='*60}")
    print(f"BEST LEARNING RATE: {best_result['lr']:.2e}")
    print(f"Best Val AUC: {best_result['best_val_auc']:.4f}")
    print(f"Best Val Loss: {best_result['best_val_loss']:.4f}")
    print(f"{'='*60}\n")
    
    return results


def main_lr_finder(device='cuda', weight_decay=1e-3, variant='MLP'):
    """Run learning rate finder (fast method)"""
    print("Starting Learning Rate Finder (Range Test Method)...")
    print(f"Using Device: {device}")
    print(f"Using Weight Decay: {weight_decay}")
    print(f"Using Variant: {variant}")
    print(f"Using Custom Threshold: {CUSTOM_THRESHOLD}")
    
    # Create Config with device and variant arguments injected
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], '--device', device, '--weight_decay', str(weight_decay), '--variant', variant]
    
    conf = Config()
    
    # Restore sys.argv
    sys.argv = original_argv
    
    # Override configuration values
    conf.task_name = const.ERROR_RECOGNITION
    conf.enable_wandb = False
    conf.device = device
    conf.weight_decay = weight_decay
    conf.variant = variant
    
    print(f"Config.device: {conf.device}")
    print(f"Config.weight_decay: {conf.weight_decay}")
    print(f"Config.variant: {conf.variant}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = train_step_test_step_dataset_base(conf)
    
    # Create model
    model = fetch_model(conf)
    
    # Verify model device
    actual_device = next(model.parameters()).device
    print(f"Model is on device: {actual_device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=conf.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Create LR Finder
    lr_finder = LearningRateFinder(model, optimizer, criterion, str(device))
    
    # Run range test
    print("\nRunning learning rate range test...")
    lr_finder.range_test(
        train_loader, 
        val_loader,
        start_lr=1e-7,
        end_lr=1e-1,
        num_iter=100,
        smooth_f=0.05
    )
    
    # Plot results
    lr_finder.plot(skip_start=10, skip_end=5, suggest=True)


def main_grid_search(device='cuda', weight_decay=1e-3, variant='MLP'):
    """Run grid search over learning rates (slower but more accurate)"""
    print("Starting Learning Rate Grid Search...")
    print(f"Using Device: {device}")
    print(f"Using Weight Decay: {weight_decay}")
    print(f"Using Variant: {variant}")
    print(f"Using Custom Threshold: {CUSTOM_THRESHOLD}")
    
    # Create Config with device and variant arguments injected
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], '--device', device, '--weight_decay', str(weight_decay), '--variant', variant]
    
    conf = Config()
    
    # Restore sys.argv
    sys.argv = original_argv
    
    # Override configuration values
    conf.task_name = const.ERROR_RECOGNITION
    conf.enable_wandb = False
    conf.device = device
    conf.weight_decay = weight_decay
    conf.variant = variant
    
    print(f"Config.device: {conf.device}")
    print(f"Config.weight_decay: {conf.weight_decay}")
    print(f"Config.variant: {conf.variant}")
    
    # Define learning rates to test
    lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    # Run grid search
    results = grid_search_lr(conf, lr_values, num_epochs=5)
    
    return results


if __name__ == "__main__":
    if args.method == 'finder':
        main_lr_finder(device=args.device, weight_decay=args.weight_decay, variant=args.variant)
    else:
        main_grid_search(device=args.device, weight_decay=args.weight_decay, variant=args.variant)