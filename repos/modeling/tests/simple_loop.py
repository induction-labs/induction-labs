import torch
from functools import partial
import random
import numpy as np
import os
from tqdm import tqdm
import wandb
import logging
from modeling.modules.action_instruct.action_head_only import analytical_distance
from modeling.data.video_action import ActionDataset, RangeActionDatapackConfig, ActionDatasetArgs, make_raw_prompt
from modeling.modules.base_module import lr_lambda

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

run = wandb.init(
    project="mouse_following",
    entity="induction-labs",
    config={},
)

class SimpleTrainer:
    def __init__(self, device="cuda", dtype=torch.float32, lr=1e-3, lr_scheduler=False, warmup_steps=200):
        self.device = device
        self.dtype = dtype
        
        # Load model and data
        self.model_test = self.load_weights()
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.model_test.parameters(), lr=lr)
        if lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                partial(
                    lr_lambda,
                    warmup_steps=warmup_steps,
                    end_steps=3000,
                    start_lr=1e-3,
                    end_lr=1e-5,
                ),
            )
        else:
            self.lr_scheduler = None
        
        # Load data tensors
        self.hidden_states = torch.load("hidden_states.pt")
        self.hidden_states = torch.randn_like(self.hidden_states).to(
            device=self.device, dtype=self.dtype
        )
        self.action_tokens = torch.load("action_tokens.pt")
        
    def load_weights(self):
        """Load the model weights"""
        logger.info(f"Loading model weights to device {self.device}")
        model_test = torch.load("action_head.pt", weights_only=False)
        model_test.to(self.device, dtype=self.dtype)
        model_test.train()
        model_test.requires_grad_(True)
        return model_test
    
    def compute_loss(self, outputs_actions, inputs):
        """Compute the loss for training"""
        # Handle shape mismatch: outputs_actions is [1, 4096, 6], squeeze batch dim
        if outputs_actions.dim() == 3 and outputs_actions.shape[0] == 1:
            outputs_actions = outputs_actions.squeeze(0)  # Now [4096, 6]
        
        output_actions = (
            outputs_actions[inputs.action_tokens]
            .reshape(-1, 2, 3)
            .to(device=self.device, dtype=self.dtype)
        )
        
        # cursor_path is already [4096, 2, 3], just filter by action_tokens
        cursor_path = (
            inputs.cursor_path[inputs.action_tokens]
            .to(device=self.device, dtype=self.dtype)
        )
        
        assert output_actions.shape == cursor_path.shape, (
            f"Expected output_actions shape {output_actions.shape} to match "
            f"cursor_path shape {cursor_path.shape}"
        )
        
        # Calculate analytical distance loss
        loss = analytical_distance(
            a=output_actions[:, 0, :],
            b=cursor_path[:, 0, :],
        ) + analytical_distance(
            a=output_actions[:, 1, :],
            b=cursor_path[:, 1, :],
        )
        
        loss = loss.sum() / inputs.action_tokens.sum().clamp(min=1.0)
        return loss

        
    def train_step(self, inputs):
        """Single training step"""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model_test(self.hidden_states)
        
        # Compute loss
        loss = self.compute_loss(outputs, inputs)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        if self.lr_scheduler:
            # Step the learning rate scheduler
            self.lr_scheduler.step()
        
        return loss.item()
    
    def train(self, dataloader, num_epochs=10):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Move inputs to device if needed
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                
                # Training step
                loss = self.train_step(inputs)
                epoch_loss += loss
                num_batches += 1
                
                # Log every 10 batches
                # if batch_idx % 10 == 0:
                #     logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.6f}")
                
                wandb.log({
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "loss": loss,
                    "lr": self.optimizer.param_groups[0]['lr'],
                })
            
            # Log epoch results
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint every few epochs
            # if (epoch + 1) % 5 == 0:
            #     self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model_test.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        logger.info(f"Checkpoint saved to {filename}")

# Usage example:
if __name__ == "__main__":
    # seed = 42
    # os.environ["PYTHONHASHSEED"] = str(
    #     seed
    # )  # make hash-based operations reproducible
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
    #     ":4096:8"  # for reproducibility in cuBLAS
    # )

    # # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    # random.seed(seed)  # Python RNG
    # np.random.seed(seed)  # NumPy RNG
    # torch.manual_seed(seed)  # CPU RNG
    # torch.cuda.manual_seed(seed)  # current GPU RNG
    # torch.cuda.manual_seed_all(seed)  # all-GPU RNGs
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # # # In PyTorch ≥1.8 you can also do:
    # torch.use_deterministic_algorithms(True)
    # g = torch.Generator()
    # g.manual_seed(seed)
    # Initialize trainer

    trainer = SimpleTrainer(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16, lr_scheduler=True, warmup_steps=0)
    
    # Assuming you have a dataloader with your ActionDataSample objects
    # dataloader = your_dataloader_here
    action_data_config = RangeActionDatapackConfig(
        prefix="gs://induction-labs/jonathan/synth/cursor_follow_v3/sample_",
        end_index=5000, 
    )
    dataloader = ActionDataset(ActionDatasetArgs(
        data_paths=action_data_config.data_paths,
        max_seq_length=4096,
        frames_per_action=2,
        raw_prompt=make_raw_prompt(
            prefix="",
            suffix="",
        )
    ))
    
    # Start training
    trainer.train(dataloader, num_epochs=20)