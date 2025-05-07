import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import logging
import gc
import matplotlib
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score # Keep only if needed for any metric
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger(__name__)

class MovementTrainer:
    """Movement analysis training manager for LSTM Autoencoder."""

    def __init__(self, config):
        self.config = config  # Config object for general settings
        self.device = config.device
        self.model = None     # LSTM Autoencoder model
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            self._verify_cuda_settings()

    def _verify_cuda_settings(self) -> None:
        """Verify and optimize CUDA settings."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            gpu_props = torch.cuda.get_device_properties(self.device)
            total_memory = gpu_props.total_memory / 1024**3
            logger.info(f"GPU: {gpu_props.name}, Total Memory: {total_memory:.1f}GB")

    def _initialize_model(self):
        """Initializes the LSTM Autoencoder model."""
        try:
            logger.info("Initializing LSTM Autoencoder model")
            from models.lstm_movement import LSTMAutoencoder

            model_config = self.config.get_model_config()
            self.model = LSTMAutoencoder(**model_config).to(self.device)
        except ImportError as e:
            logger.critical(f"Failed to import LSTMAutoencoder: {e}")
            self.model = None
        except Exception as e:
            logger.critical(f"Failed to initialize autoencoder model: {e}", exc_info=True)
            self.model = None

    def train(self, train_loader, val_loader):
        """Train the LSTM Autoencoder model."""

        logger.info("Starting model training (autoencoder mode):")
        if not train_loader or not val_loader or len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
            logger.error("Training or validation loader is empty. Cannot train.")
            if self.model is None:
                self._initialize_model()  # Initialize model only if it doesn't exist
            return self.model  # Return potentially uninitialized model

        # Initialize model if not already done
        if self.model is None:
            self._initialize_model()
        if self.model is None:
            return None  # If initialization failed

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, 'l2_regularization', 1e-4)
        )

        # LR Scheduler setup (no changes needed here)
        warmup_epochs = 5
        total_epochs = self.config.num_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                phase = max(0, epoch - warmup_epochs)
                total_phase = max(1, total_epochs - warmup_epochs)
                phase = min(phase, total_phase)
                return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * phase / total_phase))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Loss function: MSE Loss for reconstruction
        criterion = nn.MSELoss()

        self.epochs = []
        self.train_losses, self.val_losses = [], []
        self.learning_rates = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = getattr(self.config, 'early_stopping_patience', 10)

        try:
            for epoch in range(self.config.num_epochs):
                epoch_num = epoch + 1
                logger.info(f"--- Epoch {epoch_num}/{self.config.num_epochs} ---")

                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                val_loss = self._validate(val_loader, criterion)

                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                self.epochs.append(epoch_num)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.learning_rates.append(current_lr)

                logger.info(
                    f"Epoch {epoch_num} Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
                )

                # --- Model Saving Logic (based on validation LOSS) ---
                improved = False
                if val_loss < best_val_loss:  # Save based on validation LOSS
                    best_val_loss = val_loss
                    improved = True
                    logger.info(
                        f"*** New best validation loss: {best_val_loss:.4f} at epoch {epoch_num} ***"
                    )

                if improved:
                    patience_counter = 0
                    try:
                        # Save checkpoint with model_config
                        torch.save(
                            {
                                'epoch': epoch_num,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'model_config': self.config.get_model_config(), # Save model config
                            },
                            self.config.model_save_path,
                        )
                        logger.info(f"Saved best model checkpoint to {self.config.model_save_path}")
                    except Exception as save_err:
                        logger.error(f"Error saving model checkpoint: {save_err}", exc_info=True)
                else:
                    patience_counter += 1
                    logger.info(
                        f"No improvement in validation loss for {patience_counter} epochs."
                    )
                    if patience_counter >= early_stopping_patience // 2 and current_lr > 1e-6:
                        logger.warning(
                            f"Reducing learning rate due to plateau (patience: {patience_counter}/{early_stopping_patience})"
                        )
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        patience_counter = max(0, patience_counter - early_stopping_patience // 3)

                    if patience_counter >= early_stopping_patience:
                        logger.warning(f"EARLY STOPPING triggered after epoch {epoch_num}")
                        break

                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # --- Post Training ---
            # The main script will load the best model using model_loader after training.
            # No need to reload here just for plotting or final return value.
            self._save_training_plots()  # Uses data stored in self attributes (Loss only now)


        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except Exception as e:
            logger.error(f"Error in training loop: {str(e)}", exc_info=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info("Training process finished.")
        return self.model # Return the model instance


    def _train_epoch(self, train_loader, optimizer, criterion):
        """Train the autoencoder for one epoch."""

        self.model.train()
        total_loss = 0
        processed_batches = 0
        pbar = tqdm(train_loader, desc="Training", leave=False, total=len(train_loader))

        for batch_idx, batch_data in enumerate(pbar):
            try:
                # Assuming train_loader yields only the input sequence (x)
                if len(batch_data) != 3:
                    logger.warning(
                        f"Skipping training batch {batch_idx}: Expected 3 items, got {len(batch_data)}"
                    )
                    continue
                data, _, job_ids = batch_data  # Unpack, assuming labels are not needed
                data = data.to(self.device)

                optimizer.zero_grad()
                reconstructed_data = self.model(data, job_ids if self.config.conditional_autoencoder else None) # Pass job_ids if conditional
                loss = criterion(reconstructed_data, data) # MSE Loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                processed_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():}.4f'})

                if batch_idx > 0 and batch_idx % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except RuntimeError as e:  # OOM Handling
                if "out of memory" in str(e).lower():
                    logger.error(f"\nGPU OOM detected in training batch {batch_idx}! Trying to recover...")
                    # OOM recovery logic (reduce batch size) - simplified
                    current_batch_size = train_loader.batch_size
                    if current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        logger.warning(
                            f"Attempting to reduce batch size from {current_batch_size} to {new_batch_size}."
                        )
                        try:
                            new_loader = torch.utils.data.DataLoader(
                                train_loader.dataset,
                                batch_size=new_batch_size,
                                shuffle=train_loader.shuffle,
                                num_workers=train_loader.num_workers,
                                pin_memory=train_loader.pin_memory,
                                drop_last=train_loader.drop_last,
                                persistent_workers=getattr(train_loader, 'persistent_workers', False)
                            )
                            logger.info("Successfully created new DataLoader with reduced batch size.")
                            pbar.close()
                            return self._train_epoch(new_loader, optimizer, criterion) # Recursive call
                        except Exception as loader_exc:
                            logger.error(f"Failed to create new DataLoader after OOM: {loader_exc}")
                            raise e
                    else:
                        logger.critical("OOM occurred even with batch size 1! Cannot recover.")
                        raise e
                else:
                    logger.error(f"Runtime error in training batch {batch_idx}: {str(e)}", exc_info=True)
                    continue  # Skip batch
            except Exception as general_exc:
                logger.error(
                    f"General error in training batch {batch_idx}: {str(general_exc)}",
                    exc_info=True
                )
                continue  # Skip batch

        pbar.close()
        avg_loss = total_loss / max(1, processed_batches)
        logger.info(f"Training Epoch complete - Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self, val_loader, criterion):
        """Validate the autoencoder."""

        self.model.eval()
        total_loss = 0
        all_losses = []  # Store individual batch losses for analysis

        if val_loader is None or len(val_loader.dataset) == 0:
            logger.warning("Validation loader is empty. Skipping validation.")
            return float('inf')

        with torch.no_grad():
            for batch_data in val_loader:
                try:
                    # Assuming val_loader yields only the input sequence (x)
                    if len(batch_data) != 3:
                        logger.warning(
                            f"Skipping validation batch: Expected 3 items, got {len(batch_data)}"
                        )
                        continue
                    data, _, job_ids = batch_data
                    data = data.to(self.device)

                    reconstructed_data = self.model(data, job_ids if self.config.conditional_autoencoder else None)
                    loss = criterion(reconstructed_data, data)

                    total_loss += loss.item()
                    all_losses.append(loss.item())

                except Exception as e:
                    logger.error(f"Error processing validation batch: {e}", exc_info=True)
                    continue

        avg_loss = total_loss / max(1, len(val_loader))
        logger.info(f"Validation Complete - Loss: {avg_loss:.4f}")
        return avg_loss

    def _save_training_plots(self):
        """Save training history plots (reconstruction loss only)."""

        viz_dir = self.config.output_dir / 'visualization'
        viz_dir.mkdir(parents=True, exist_ok=True)
        try:
            epochs = self.epochs
            if not epochs:
                logger.warning("No epoch data to plot.")
                return

            plt.figure(figsize=(12, 6))
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
            if self.val_losses:
                best_epoch_idx = np.nanargmin(self.val_losses)
                plt.plot(
                    epochs[best_epoch_idx],
                    self.val_losses[best_epoch_idx],
                    'r*',
                    markersize=12,
                    label=f'Best Val Loss: {self.val_losses[best_epoch_idx]:.4f}',
                )
            plt.title('Training and Validation Reconstruction Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Reconstruction Loss (MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            loss_path = viz_dir / 'autoencoder_loss_history.png'  # Changed filename
            plt.savefig(loss_path, dpi=150)
            plt.close()
            logger.info(f"Reconstruction loss history plot saved: {loss_path}")

            if self.learning_rates:  # Learning Rate
                plt.figure(figsize=(12, 6))
                plt.plot(epochs, self.learning_rates, 'c.-', label='Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                lr_path = viz_dir / 'learning_rate_schedule.png'
                plt.savefig(lr_path, dpi=150)
                plt.close()
                logger.info(f"Learning rate schedule plot saved: {lr_path}")

        except Exception as e:
            logger.error(f"Error creating training plots: {str(e)}", exc_info=True)
            plt.close('all')