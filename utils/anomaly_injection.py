import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import os
            
def sample_anomaly_params():
    """
    Samples anomaly parameters (A, B, C) for one sample.
    Ensures the anomaly stays within realistic bounds.
    """
    B = 0.385
    max_val = 2.0
    max_at_t30 = 0.4
    scale_factor = 90409

    while True:
        A = np.random.normal(loc=74120, scale=20000)
        C = np.random.normal(loc=0.806, scale=0.2)

        days = np.arange(1, 61)  # Avoid day 0
        values = (A * days * np.exp(-B * (days ** C))) / scale_factor

        if (
            np.all(values <= max_val) and
            values[29] < max_at_t30 and
            np.all(values >= 0)
        ):
            return A, B, C
        

def inject_continuous_anomalies(
    seq_x_batch, seq_y_batch,
    seq_len, label_len, pred_len,
    scaler,
    ano_type='input_to_output'  # or 'input_only', or 'none'
):
    """
    Injects continuous anomalies with batch-consistent shape parameters.
    Injection location is determined by `ano_type`.

    Args:
        ano_type: 'input_only' or 'input_to_output'

    Returns:
        Tuple of (seq_x_anom, seq_y_anom)
    """
    if scaler is None:
        raise ValueError("A valid `scaler` instance must be provided for inverse-transform and re-scaling.")

    B_size, F = seq_x_batch.shape[0], seq_x_batch.shape[2]
    new_seq_x = []
    new_seq_y = []

    # Shared anomaly shape parameters for the batch
    A_batch, B_param, C_batch = sample_anomaly_params()
    A_std = 0.1 * A_batch
    C_std = 0.05 * C_batch

    total_len = seq_len + pred_len
    scale_factor = 90409

    # Set injection bounds based on type
    if ano_type == 'input_only':
        min_offset = int(seq_len * 0.0)
        max_offset = int(seq_len * 0.5)
    elif ano_type == 'input_to_output':
        min_offset = int(seq_len * 0.85) 
        max_offset = int(seq_len * 0.95) 
    else:
        # Skip injection and return original
        return seq_x_batch.copy(), seq_y_batch.copy()

    for b in range(B_size):
        seq_x = seq_x_batch[b]
        seq_y = seq_y_batch[b]

        # De-normalize
        seq_x_orig = scaler.inverse_transform(seq_x)
        seq_y_orig = scaler.inverse_transform(seq_y)
        
        full_sequence_before = np.concatenate([seq_x_orig, seq_y_orig[-pred_len:]], axis=0).copy()

        full_sequence = np.concatenate([seq_x_orig, seq_y_orig[-pred_len:]], axis=0)

        # Random start in specified input region
        anomaly_start = np.random.randint(min_offset, max_offset)
        anomaly_days = np.arange(total_len - anomaly_start)

        # Per-sample variation
        A_i = np.random.normal(loc=A_batch, scale=A_std)
        C_i = np.random.normal(loc=C_batch, scale=C_std)

        # Anomaly curve
        anomaly_curve = (A_i * anomaly_days * np.exp(-B_param * (anomaly_days ** C_i))) / scale_factor
        sequence_mean = full_sequence.mean()
        anomaly_values = sequence_mean * anomaly_curve.reshape(-1, 1)

        # Inject
        full_sequence[anomaly_start:] += anomaly_values

        # Re-split and normalize
        seq_x_anom = full_sequence[:seq_len]
        seq_y_anom = full_sequence[-(label_len + pred_len):]
        
         # ---------- Save plots ----------
        """    
        if random.random() < 0.001:  # only save ~0.1% for efficiency
            plt.figure(figsize=(12, 5))
            plt.plot(full_sequence_before, label="Original Sequence", color="gray")
            plt.plot(full_sequence, label="Anomalous Sequence", color="red")
            plt.plot(
                range(anomaly_start, total_len),
                anomaly_values.squeeze(),
                label="Anomaly Injected",
                linestyle="--",
                color="blue"
            )
            plt.axvline(anomaly_start, color='black', linestyle=':', label="Anomaly Start")
            plt.title(f"Anomaly Injected (sample {b})")
            plt.xlabel("Time Step")
            plt.ylabel("Scaled Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the figure
            plot_dir = "/path_to_save_plots_of_injections/"
            filename = os.path.join(plot_dir, f"sample_{b}_seed{random.randint(1000, 9999)}.png")
            plt.savefig(filename)
            plt.close()
        """    
        # ----------------------------------

        seq_x_anom = scaler.transform(seq_x_anom)
        seq_y_anom = scaler.transform(seq_y_anom)

        new_seq_x.append(seq_x_anom)
        new_seq_y.append(seq_y_anom)

    return np.stack(new_seq_x), np.stack(new_seq_y)


def inject_pointwise_anomalies(
    seq_x_batch, seq_y_batch,
    seq_len, 
    label_len, 
    pred_len,
    scaler,
    ano_type ='const', 
    ano_ratio=0.1, 
    ano_scale_const=0.5, 
    ano_scale_gaussian=2
):
    """
    Injects pointwise anomalies into input and output sequences in a batch.
    Anomaly type can be 'const', 'gaussian', or 'missing'.
    Operates in normalized space (no inverse-transform).

    Parameters:
        seq_x_batch: np.ndarray of shape (B, seq_len, D)
        seq_y_batch: np.ndarray of shape (B, pred_len, D)
        seq_len: int
        label_len: int
        pred_len: int
        scaler: unused, kept for interface compatibility
        ano_type: str, one of ['const', 'gaussian', 'missing', 'none']
        anom_ratio: float, probability per time step to inject anomaly
        anom_scale_const: float, value for constant anomalies
        anom_scale_gaussian: float, stddev for gaussian anomalies

    Returns:
        Tuple (seq_x_anom, seq_y_anom), both of shape like inputs
    """
    B, T, D = seq_x_batch.shape
    new_seq_x = []
    new_seq_y = []

    np.random.seed(0)  # For reproducibility (optional)

    for b in range(B):
        seq_x = seq_x_batch[b]
        seq_y = seq_y_batch[b]
        full_sequence = np.concatenate([seq_x, seq_y[label_len:]], axis=0)
        total_len = full_sequence.shape[0]

        # Create anomaly mask
        anomaly_mask = np.random.rand(total_len) < ano_ratio

        if ano_type == 'const':
            noise = np.full((total_len, D), fill_value=ano_scale_const)
        elif ano_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=ano_scale_gaussian, size=(total_len, D))
        elif ano_type == 'missing':
            noise = -full_sequence + 0.0001  # Push to near zero
        elif ano_type == 'none':
            noise = np.zeros_like(full_sequence)
        else:
            raise ValueError(f"Unknown ano_type: {ano_type}")

        # Apply anomaly only where mask is True
        full_sequence[anomaly_mask] += noise[anomaly_mask]

        # Split back
        seq_x_anom = full_sequence[:seq_len]
        seq_y_anom = np.concatenate([seq_y[:label_len], full_sequence[seq_len:]])

        new_seq_x.append(seq_x_anom)
        new_seq_y.append(seq_y_anom)

    return np.stack(new_seq_x), np.stack(new_seq_y)