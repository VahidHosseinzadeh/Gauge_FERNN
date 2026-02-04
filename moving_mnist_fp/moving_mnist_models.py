import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class DiffLucasKanade(nn.Module):
    """
    Differentiable optical flow via exhaustive search (template matching).
    Works correctly for Moving MNIST with large motions (v_range 1-3).
    Returns PER-SAMPLE velocities (B, num_v) - NOT averaged over batch!
    """
    def __init__(self, v_range=3, smooth=0.1):
        super().__init__()
        self.v_range = v_range
        self.smooth = smooth  # Temperature for soft-argmax
        
        # Pre-compute velocity list
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) 
                      for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        
        # Register as buffer for device placement
        self.register_buffer('vel_tensor', 
                           torch.tensor(self.v_list, dtype=torch.float32))
        
    def forward(self, f_t, f_t_prev):
        """
        Args:
            f_t: (B, C, H, W) - current frame
            f_t_prev: (B, C, H, W) - previous frame
            
        Returns:
            probs: (B, num_v) - velocity probabilities per sample
        """
        B, C, H, W = f_t.shape
        
        # Pre-allocate error tensor
        errors = torch.zeros(B, self.num_v, device=f_t.device)
        
        # Compute MSE for all velocities - INDEPENDENTLY PER BATCH
        for i, (vx, vy) in enumerate(self.v_list):
            # Warp previous frame by this velocity
            f_warp = torch.roll(f_t_prev, shifts=(vy, vx), dims=(-2, -1))
            # Compute per-sample MSE: (B,)
            errors[:, i] = torch.sum((f_t - f_warp) ** 2, dim=(1, 2, 3))
        
        # Convert errors to weights via softmax (soft-argmin)
        # -errors / smooth: lower error → higher logit
        probs = F.softmax(-errors / self.smooth, dim=1)  # (B, num_v)
        
        
        return probs  # (B, num_v) per sample
    

class ParametricVelPrediction(nn.Module):
    """
    Pure discrete velocity predictor using hard classification.
    Outputs integer velocities for use with torch.roll.
    """
    def __init__(self, input_channels, v_range=2, hidden_dim=6):
        """
        Args:
            input_channels: Number of input channels in frames
            v_range: Velocity range (creates grid from -v_range to +v_range)
            hidden_dim: Hidden dimension in the network
        """
        super().__init__()

        # Create discrete velocity grid (same as FERNN)
        # Generate all integer velocity pairs in the range
        self.v_range = v_range
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1)
                      for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        # Convert v_list to tensor for easy indexing
        self.register_buffer('velocity_tensor',
                           torch.tensor(self.v_list, dtype=torch.long))

        # Feature extractor: takes two frames, outputs features
        # Uses spatial conv to preserve location information for motion
        self.feature_extractor = nn.Sequential(
            # Input: concatenated frames [f_t, f_t_prev]
            nn.Conv2d(2 * input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Spatial velocity classifier: outputs per-pixel logits, then pools
        # This preserves spatial info during feature extraction
        self.velocity_conv = nn.Conv2d(hidden_dim, self.num_v, 3, padding=1)

    def forward(self, f_t, f_t_prev):
        """
        Predicts discrete velocity between two frames using hard classification.

        Args:
            f_t: (batch, C, H, W) - current frame
            f_t_prev: (batch, C, H, W) - previous frame

        Returns:
           probs: (batch, num_v) - softmax probabilities (differentiable) over velocities
        """
        batch_size = f_t.shape[0]

        # Step 1: Concatenate frames along channel dimension
        # Shape: (batch, 2*C, H, W)
        x = torch.cat([f_t, f_t_prev], dim=1)

        # Step 2: Extract spatial features
        # Shape: (batch, hidden_dim, H, W)
        features = self.feature_extractor(x)

        # Step 3: Get per-pixel velocity logits, then pool
        # Shape: (batch, num_v, H, W) -> (batch, num_v)
        logits = self.velocity_conv(features).mean(dim=(2, 3))

        # Step 4: Softmax probabilities (differentiable)
        probs = F.softmax(logits, dim=1)

        # Step 5: Hard classification - take argmax
        # Returns index of highest logit for each batch
        # Shape: (batch,)
        indices = torch.argmax(logits, dim=1)

        # Step 6: Convert indices to velocity vectors
        # Use advanced indexing: velocity_tensor[indices] gives (batch, 2)
        u_t = self.velocity_tensor[indices]  # Shape: (batch, 2)

        return probs











class FERNN_Cell(nn.Module):
    """
    RNN cell with discrete velocity and integer shifts.
    Implements: h_{t+1}(x) = ψ₁(u_t) · [W ⋆ h_t](x) + E[f_t](x)
    Uses torch.roll for integer velocity shifts.
    """
    def __init__(self, input_channels, hidden_channels, 
                 h_kernel_size=3, u_kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Convolution W for hidden state: W ⋆ h_t
        h_pad = h_kernel_size // 2
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                padding=h_pad, padding_mode='circular', bias=False)
        
        # Encoder E (convolution U) for input: U ⋆ f_t
        u_pad = u_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                padding=u_pad, padding_mode='circular', bias=False)
        
        # Activation function
        self.activation = nn.ReLU()


    def apply_flow_soft(self, h, probs, v_list):
        """
        Differentiable flow: weighted sum of all discrete shifts.
        
        Args:
            h: (batch, C, H, W) - tensor to warp
            probs: (batch, num_v) - softmax over velocities
            v_list: list of (dx, dy) tuples
        Returns:
            warped_h: (batch, C, H, W)
        """
        # batch, C, H, W = h.shape
        
        # warped = torch.zeros_like(h)

        # for idx, (dx, dy) in enumerate(v_list):
        #     shifted = torch.roll(h, shifts=(dy, dx), dims=(2, 3))
        #     w = probs[:, idx].view(batch, 1, 1, 1)
        #     warped = warped + w * shifted

        
    
        B, C, H, W = h.shape
        V = len(v_list)

        # stack all shifted versions: (V, B, C, H, W)
        shifted = torch.stack(
            [
                torch.roll(h, shifts=(dy, dx), dims=(2, 3))
                for (dx, dy) in v_list
            ],
            dim=0
        )

        # reshape probs for broadcasting: (V, B, 1, 1, 1)
        probs = probs.transpose(0, 1).view(V, B, 1, 1, 1)

        # weighted sum over velocities
        warped_h = (probs * shifted).sum(dim=0)

        return warped_h
    
    # this might be better as the gradient is not just 1 I think compared to roll function 
    def apply_flow_differentiable(self, x, probs, v_list):
        """
        Differentiable warping using grid_sample.
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create base grid
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, W, device=device),
            torch.linspace(-1, 1, H, device=device),
            indexing='ij'
        ), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        warped = torch.zeros_like(x)
        
        for idx, (dx, dy) in enumerate(v_list):
            # Create flow field for this velocity
            flow_x = (2 * dx / W) * torch.ones_like(base_grid[..., 0:1])
            flow_y = (2 * dy / H) * torch.ones_like(base_grid[..., 1:2])
            flow = torch.cat([flow_x, flow_y], dim=-1)
            
            # Warp using grid_sample
            grid = base_grid + flow
            shifted = F.grid_sample(
                x, grid,
                mode='bilinear', padding_mode='border',
                align_corners=True
            )
            
            # Weighted sum
            w = probs[:, idx].view(B, 1, 1, 1)
            warped = warped + w * shifted
    
        return warped
    

    
    def forward(self, f_t, h_t,  probs=None, v_list=None, alpha=1.0, use_differentiable_flow=False):
        """
        Forward pass of the RNN cell.
        
        Args:
            f_t: (batch, input_channels, H, W) - current input frame
            h_t: (batch, hidden_channels, H, W) - current hidden state
            probs: (batch, num_v) - softmax over velocities (optional)
            v_list: list of (dx, dy) tuples
            alpha: float - annealing parameter for soft flow (0 to 1) how much I want to warp h
        Returns:
            h_next: (batch, hidden_channels, H, W) - next hidden state
        """
        # Step 1: Convolve hidden state: [W ⋆ h_t]
        conv_h = self.conv_h(h_t)  # (batch, hidden_channels, H, W)
        
        # Step 2: Apply flow transformation: ψ₁(u_t) · [W ⋆ h_t]   
        # we can aneal the alpha from 0 to 1 during training
        if use_differentiable_flow:
            warped_conv_h = (1 - alpha) * conv_h + alpha * self.apply_flow_differentiable(conv_h, probs, v_list)
        else:
            warped_conv_h = (1 - alpha) * conv_h + alpha * self.apply_flow_soft(conv_h, probs, v_list)
        encoded_f = self.conv_u(f_t)  # (batch, hidden_channels, H, W)
        h_next = self.activation(warped_conv_h + encoded_f)
        
        return h_next

class Seq2SeqFERNN(nn.Module):
    """
    Complete sequence-to-sequence model with discrete velocity prediction.
    """
    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=3, decoder_conv_layers=1,pool_type='max', use_differentiable_flow=True):
        """
        Args:
            input_channels: Number of channels in input frames
            hidden_channels: Number of channels in hidden state
            height, width: Spatial dimensions of input
            output_channels: Number of output channels (defaults to input_channels)
            h_kernel_size: Kernel size for hidden state convolution
            u_kernel_size: Kernel size for input encoding
            v_range: Range of discrete velocities (creates (2*v_range+1)^2 velocities)
            decoder_conv_layers: Number of convolutional layers in decoder
        """
        super().__init__()
        self.pool_type = pool_type
        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels
        self.v_range = v_range
        self.use_differentiable_flow = use_differentiable_flow
        
        # Discrete velocity predictor (hard classification)
        self.velocity_predictor = ParametricVelPrediction( input_channels=1, v_range=2, hidden_dim=6)
        
        # Main RNN cell (uses integer shifts)
        self.cell = FERNN_Cell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size

        )
        
        # Decoder: converts hidden state to output frame
        decoder_layers = []
        for _ in range(decoder_conv_layers):
            decoder_layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, 3,
                         padding=1, padding_mode='circular', bias=False),
                nn.ReLU()
            ])
        decoder_layers.append(
            nn.Conv2d(hidden_channels, self.output_channels, 3,
                     padding=1, padding_mode='circular', bias=False)
        )
        self.decoder = nn.Sequential(*decoder_layers)
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.cell.hidden_channels,
                          self.height, self.width, device=device)
    
    def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0,
                target_seq=None, return_vel_probs=False):
        """
        Forward pass for sequence prediction.
        
        Args:
            input_seq: (batch, T_in, C, H, W) - input sequence
            pred_len: int - number of frames to predict
            teacher_forcing_ratio: float - probability of using ground truth
            target_seq: (batch, pred_len, C, H, W) - ground truth for teacher forcing
            return_vel_probs: bool - whether to return velocity probability distributions
            
        Returns:
            predictions: (batch, pred_len, C, H, W) - predicted frames
            vel_probs: optional (batch, T_in+pred_len-1, num_v) - velocity probability distributions
        """
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # Initialize hidden state
        h = self.init_hidden(batch, device)
        
        # Store velocities if requested
        vel_probs = []
        
        # --- Encoder Phase: Process input sequence ---
        for t in range(T_in):
            f_t = input_seq[:, t]  # Current frame
            
            # Estimate velocity (need current and previous frame)
            if t == 0:
                # First frame: no previous frame, use zero velocity
                probs = torch.zeros(batch, self.velocity_predictor.num_v, device=device)
                probs[:, self.velocity_predictor.num_v // 2] = 1.0
            else:
                f_t_prev = input_seq[:, t-1]  # Previous frame
                probs = self.velocity_predictor(f_t, f_t_prev)
            
            # Update RNN (soft during training, hard during eval)
            h = self.cell(
                f_t,
                h,
                probs=probs,
                v_list=self.velocity_predictor.v_list,
                alpha=1.0 if not self.training else min(1.0, t / T_in),
                use_differentiable_flow = self.use_differentiable_flow
            )
            
            # Store velocity
            if return_vel_probs:
                vel_probs.append(probs)
        
        # --- Decoder Phase: Generate predictions ---
        prev_frame = input_seq[:, -1]  # Start with last input frame
        predictions = []
        
        for t in range(pred_len):
            # Determine current frame (teacher forcing or previous prediction)
            if self.training and target_seq is not None and torch.rand(1) < teacher_forcing_ratio:
                current_frame = target_seq[:, t]
            else:
                current_frame = prev_frame
            
            # Estimate velocity for this step
            if t == 0:
                # First prediction: use last real frame as previous
                f_t_prev = input_seq[:, -1]
            else:
                # Use previous prediction as previous frame
                f_t_prev = predictions[-1]
            
            probs = self.velocity_predictor(current_frame, f_t_prev)
            
            # Update RNN (soft during training, hard during eval)
            h = self.cell(
                current_frame,
                h,
                probs=probs,
                v_list=self.velocity_predictor.v_list,
                alpha=1.0 if not self.training else min(1.0, (T_in + t) / (T_in + pred_len))
            )
            
            # Decode hidden state to frame prediction
            pred = self.decoder(h)
            predictions.append(pred)
            prev_frame = pred  # For next iteration
            
            # Store velocity
            if return_vel_probs:
                vel_probs.append(probs)
        
        # Stack predictions along time dimension
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_len, C, H, W)
        
        if return_vel_probs:
            vel_probs = torch.stack(vel_probs, dim=1)  # (batch, T_in+pred_len-1, num_v)
            return predictions, vel_probs
        else:
            return predictions




# class FERNN_Cell(nn.Module):
#     def __init__(self, input_channels, hidden_channels,
#                  h_kernel_size=3, u_kernel_size=3, v_range=0):
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
#         self.num_v = len(self.v_list)

#         # circular convs without bias
#         u_pad = u_kernel_size // 2
#         h_pad = h_kernel_size // 2
#         self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
#                                  padding=u_pad, padding_mode='circular', bias=False)
#         self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
#                                  padding=h_pad, padding_mode='circular', bias=False)
#         self.activation = nn.ReLU()

#     def forward(self, u_t, h_prev):
#         # u_t: (batch, C, H, W)
#         # h_prev: (batch, num_v, hidden, H, W)
#         batch, C, H, W = u_t.size()
#         # conv_u then expand
#         u_conv = self.conv_u(u_t)  # (batch, hidden, H, W)
#         u_conv = u_conv.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

#         # shift hidden via torch.roll per velocity
#         h_shift = []
#         for i, (vx, vy) in enumerate(self.v_list):
#             h_shift.append(torch.roll(h_prev[:, i], shifts=(vy, vx), dims=(2, 3)))
#         h_shift = torch.stack(h_shift, dim=1)  # (batch, num_v, hidden, H, W)

#         # conv_h on flattened v dimension
#         h_flat = h_shift.view(batch * self.num_v, self.hidden_channels, H, W)
#         h_conv = self.conv_h(h_flat)
#         h_conv = h_conv.view(batch, self.num_v, self.hidden_channels, H, W)

#         # combine and activate
#         h_next = self.activation(u_conv + h_conv)
#         return h_next

# class Seq2SeqFERNN(nn.Module):
#     def __init__(self, input_channels, hidden_channels, height, width,
#                  output_channels=None, h_kernel_size=3, u_kernel_size=3,
#                  v_range=0, pool_type='max', decoder_conv_layers=1):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.pool_type = pool_type
#         self.output_channels = output_channels or input_channels

#         self.cell = FERNN_Cell(
#             input_channels, hidden_channels,
#             h_kernel_size, u_kernel_size, v_range)
#         self.hidden_channels = hidden_channels
#         self.num_v = self.cell.num_v

#         decoder = []
#         for _ in range(decoder_conv_layers):
#             decoder += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='circular', bias=False), nn.ReLU()]
#         decoder += [nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1, padding_mode='circular', bias=False)]
#         self.decoder_conv = nn.Sequential(*decoder)

#     def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0, target_seq=None, return_hidden=False):
#         batch, T_in, C, H, W = input_seq.size()
#         device = input_seq.device

#         if return_hidden:
#             input_seq_hiddens = torch.zeros(batch, T_in, self.num_v, self.hidden_channels, H, W, device=device)
#             out_seq_hiddens = torch.zeros(batch, pred_len, self.num_v, self.hidden_channels, H, W, device=device)

#         # Initialize hidden state
#         h = torch.zeros(batch, self.num_v, self.hidden_channels, H, W, device=device)

#         # Encoder pass through cell
#         for t in range(T_in): 
#             u_t = input_seq[:, t]
#             h = self.cell(u_t, h)

#             if return_hidden:
#                 input_seq_hiddens[:, t] += h.detach()

#         prev = input_seq[:, -1]
#         outputs = []

#         # Decoder
#         for t in range(pred_len):
#             if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
#                 frame = target_seq[:, t]
#             else:
#                 frame = prev.detach()
#             h = self.cell(frame, h)

#             if return_hidden:
#                 out_seq_hiddens[:, t] += h.detach()

#             # pool over velocities
#             if self.pool_type == 'max':
#                 feat = h.max(1)[0]
#             elif self.pool_type == 'mean':
#                 feat = h.mean(1)
#             elif self.pool_type == 'sum':
#                 feat = h.sum(1)
#             else:
#                 feat = h.max(1)[0]

#             out = self.decoder_conv(feat)
#             outputs.append(out)
#             prev = out

#         if return_hidden:
#             return torch.stack(outputs, dim=1), input_seq_hiddens, out_seq_hiddens # _, (B, T_in, num_v, C, H, W), (B, T_out, num_v, C, H, W)
#         else:
#             return torch.stack(outputs, dim=1)
