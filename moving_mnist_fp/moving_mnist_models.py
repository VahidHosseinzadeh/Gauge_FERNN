import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F


# class DiffLucasKanade(nn.Module):
#     """
#     Differentiable optical flow via exhaustive search (template matching).
#     Works correctly for Moving MNIST with large motions (v_range 1-3).
#     Returns PER-SAMPLE velocities (B, num_v) - NOT averaged over batch!
#     """
#     def __init__(self, v_range=3, smooth=0.1):
#         super().__init__()
#         self.v_range = v_range
#         self.smooth = smooth  # Temperature for soft-argmax
        
#         # Pre-compute velocity list
#         self.v_list = [(x, y) for x in range(-v_range, v_range + 1) 
#                       for y in range(-v_range, v_range + 1)]
#         self.num_v = len(self.v_list)
        
#         # Register as buffer for device placement
#         self.register_buffer('vel_tensor', 
#                            torch.tensor(self.v_list, dtype=torch.float32))
        
#     def forward(self, f_t, f_t_prev):
#         """
#         Args:
#             f_t: (B, C, H, W) - current frame
#             f_t_prev: (B, C, H, W) - previous frame
            
#         Returns:
#             probs: (B, num_v) - velocity probabilities per sample
#         """
#         B, C, H, W = f_t.shape
        
#         # Pre-allocate error tensor
#         errors = torch.zeros(B, self.num_v, device=f_t.device)
        
#         # Compute MSE for all velocities - INDEPENDENTLY PER BATCH
#         for i, (vx, vy) in enumerate(self.v_list):
#             # Warp previous frame by this velocity
#             f_warp = torch.roll(f_t_prev, shifts=(vy, vx), dims=(-2, -1))
#             # Compute per-sample MSE: (B,)
#             errors[:, i] = torch.sum((f_t - f_warp) ** 2, dim=(1, 2, 3))
        
#         # Convert errors to weights via softmax (soft-argmin)
#         # -errors / smooth: lower error → higher logit
#         probs = F.softmax(-errors / self.smooth, dim=1)  # (B, num_v)
        
        
#         return probs  # (B, num_v) per sample
    


class ParametricVelPrediction(nn.Module):
    """
    Improved discrete velocity predictor using hard classification.
    Enhanced with attention, multi-scale features, and explicit motion cues.
    """
    def __init__(self, input_channels, v_range=3, hidden_dim=64, use_attention=True, 
                    use_motion_diff=True, use_multiscale=True):
        """
        Args:
            input_channels: Number of input channels in frames
            v_range: Velocity range (creates grid from -v_range to +v_range)
            hidden_dim: Hidden dimension in the network
            use_attention: Whether to use channel attention (more powerful)
            use_motion_diff: Whether to compute frame differences as explicit motion cue
            use_multiscale: Whether to use multi-scale feature extraction
        """
        super().__init__()

        # Create discrete velocity grid
        self.v_range = v_range
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1)
                        for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        
        self.register_buffer('velocity_tensor',
                            torch.tensor(self.v_list, dtype=torch.long))
        
        self.use_attention = use_attention
        self.use_motion_diff = use_motion_diff
        self.use_multiscale = use_multiscale


        self.activation = nn.ReLU()

        # Input channels: 2*input_channels + (input_channels if use_motion_diff)
        feat_input_channels = 2 * input_channels
        if use_motion_diff:
            feat_input_channels += input_channels

        # Feature extractor with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(feat_input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks for better feature learning
        self.res_block1 = self._make_residual_block(hidden_dim)
        self.res_block2 = self._make_residual_block(hidden_dim)
        
        # Channel attention mechanism (optional but effective)
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Multi-scale feature extraction (optional)
        if use_multiscale:
            self.scale_pool3 = nn.MaxPool2d(3, stride=1, padding=1)
            self.scale_fusion = nn.Conv2d(2 * hidden_dim, hidden_dim, 1)
        
        # Velocity classification head with larger receptive field
        self.velocity_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.num_v, 1),
        )

    def _make_residual_block(self, channels):
        """Create a residual block for improved gradient flow."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, f_t, f_t_prev):
        """
        Predicts discrete velocity between two frames.

        Args:
            f_t: (batch, C, H, W) - current frame
            f_t_prev: (batch, C, H, W) - previous frame

        Returns:
            u_t: (batch, 2) - integer velocity (dy, dx)
            logits: (batch, num_v) - classification logits
            probs: (batch, num_v) - softmax probabilities
        """
        batch_size = f_t.shape[0]

        # Compute explicit motion cue (frame difference)
        if self.use_motion_diff:
            motion_diff = f_t - f_t_prev
            x = torch.cat([f_t, f_t_prev, motion_diff], dim=1)
        else:
            x = torch.cat([f_t, f_t_prev], dim=1)

        # Initial feature extraction
        features = self.feature_extractor(x)

        # Residual connections for better gradient flow
        residual = features
        features = self.activation(self.res_block1(features) + residual)
        residual = features
        features = self.activation(self.res_block2(features) + residual)

        # Channel attention weighting (amplifies important channels)
        if self.use_attention:
            attn_weights = self.channel_attention(features)
            features = features * attn_weights

        # Multi-scale feature fusion (captures motion at different scales)
        if self.use_multiscale:
            scale_features = self.scale_pool3(features)
            fused_features = torch.cat([features, scale_features], dim=1)
            fused_features = self.scale_fusion(fused_features)
            features = features + fused_features
            logits = self.velocity_head(features)
        else:
            logits = self.velocity_head(features)

        # Pool spatial dimensions to get per-class logits
        # Use adaptive pool for robust reduction
        logits = F.adaptive_avg_pool2d(logits, 1).view(batch_size, self.num_v)

        # Softmax probabilities (differentiable for training)
        probs = F.softmax(logits, dim=1)

        # Hard classification via argmax
        indices = torch.argmax(logits, dim=1)
        u_t = self.velocity_tensor[indices]

        return u_t, logits, probs


class FERNN_Cell(nn.Module):
    """
    Enhanced RNN cell with flow equivariance and probabilistic velocity warping.
    Improvements:
    - Efficient soft flow with better gradient propagation
    - Velocity-conditioned convolutions
    - Multi-scale processing for hierarchical motion
    - Skip connections and improved normalization
    """
    def __init__(self, input_channels, hidden_channels, 
                    h_kernel_size=3, u_kernel_size=3, 
                    use_velocity_gating=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_velocity_gating = use_velocity_gating
        
        # Convolution W for hidden state: W ⋆ h_t
        h_pad = h_kernel_size // 2
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                padding=h_pad, padding_mode='circular', bias=True)
        self.norm_h = nn.BatchNorm2d(hidden_channels)
        
        # Encoder E (convolution U) for input: U ⋆ f_t
        u_pad = u_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                padding=u_pad, padding_mode='circular', bias=True)
        self.norm_u = nn.BatchNorm2d(hidden_channels)

        
        # Velocity-conditional gating (optional: makes warping adaptive)
        if use_velocity_gating:
            self.velocity_gate = nn.Sequential(
                nn.Linear(1, 64),  # 1 input: entropy of velocity distribution
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        
        self.activation = nn.ReLU()

    def apply_flow_soft_efficient(self, h, probs, v_list):
        """
        Efficient differentiable flow using vectorized operations.
        
        Key improvements over original:
        - Vectorized shifts instead of per-sample loops
        - Better numerical stability
        - Cleaner gradient computation
        
        Args:
            h: (batch, C, H, W) - tensor to warp
            probs: (batch, num_v) - softmax over velocities
            v_list: list of (dx, dy) tuples
        Returns:
            warped_h: (batch, C, H, W)
        """
        B, C, H, W = h.shape
        V = len(v_list)
        
        # Pre-allocate output with proper device handling
        warped_h = torch.zeros_like(h)
        
        # Reshape probs for efficient broadcasting: (B, V, 1, 1, 1)
        probs_reshaped = probs.view(B, V, 1, 1, 1)
        
        # Apply each velocity shift and accumulate weighted result
        for idx, (dx, dy) in enumerate(v_list):
            shifted = torch.roll(h, shifts=(dy, dx), dims=(2, 3))
            warped_h.add_(shifted * probs_reshaped[:, idx], alpha=1.0)
        
        return warped_h

    def apply_flow_soft_batch(self, h, probs, v_list):
        """
        Alternative batch-stacking version (original implementation style).
        Use this if VRAM is not a constraint and you want maximum efficiency.
        
        Args:
            h: (batch, C, H, W)
            probs: (batch, num_v)
            v_list: list of (dx, dy) tuples
        Returns:
            warped_h: (batch, C, H, W)
        """
        B, C, H, W = h.shape
        V = len(v_list)

        # Stack all shifted versions: (V, B, C, H, W)
        shifted = torch.stack([
            torch.roll(h, shifts=(dy, dx), dims=(2, 3))
            for (dx, dy) in v_list
        ], dim=0)

        # Reshape probs for broadcasting: (V, B, 1, 1, 1)
        probs_t = probs.t().view(V, B, 1, 1, 1)

        # Weighted sum: more stable numerically
        warped_h = (probs_t * shifted).sum(dim=0)

        return warped_h

    def apply_flow_continuous(self, x, probs, v_list):
        """
        Continuous differentiable warping using grid_sample.
        Provides smoother gradients than integer shifts.
        
        Args:
            x: (batch, C, H, W)
            probs: (batch, num_v) - softmax weights
            v_list: list of (dx, dy) tuples
        Returns:
            warped_x: (batch, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Compute weighted velocity (expected displacement)
        # v_list is [(dx, dy), ...], convert to tensor
        v_tensor = torch.tensor(v_list, dtype=x.dtype, device=device)  # (V, 2)
        weighted_v = (probs @ v_tensor)  # (B, 2)
        
        # Create normalized coordinate grid for grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # Apply learned velocity to grid
        # Convert pixel displacement to normalized grid displacement
        flow_x = (2.0 * weighted_v[:, 0] / W).view(B, 1, 1)
        flow_y = (2.0 * weighted_v[:, 1] / H).view(B, 1, 1)
        
        grid_warped = grid.clone()
        grid_warped[..., 0] += flow_x
        grid_warped[..., 1] += flow_y
        
        # Apply warping with bilinear interpolation
        warped_x = F.grid_sample(
            x, grid_warped,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped_x

    def forward(self, f_t, h_t, probs=None, v_list=None, 
                use_continuous_flow=False):
        """
        Forward pass with enhanced flow warping.
        
        Args:
            f_t: (batch, input_channels, H, W) - current input frame
            h_t: (batch, hidden_channels, H, W) - current hidden state
            probs: (batch, num_v) - softmax over velocities
            v_list: list of (dx, dy) tuples
            use_continuous_flow: bool - use grid_sample instead of roll
        
        Returns:
            h_next: (batch, hidden_channels, H, W) - next hidden state
        """

        if probs is not None and v_list is not None:
            assert len(v_list) == probs.shape[1], \
            f"v_list length ({len(v_list)}) doesn't match probs shape ({probs.shape[1]})"
            
            # Step 1: MEASURE VELOCITY CONFIDENCE
        if self.use_velocity_gating and probs is not None:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
            entropy = entropy / np.log(len(v_list))  # Normalize to [0, 1]
            confidence = self.velocity_gate(entropy)  # (batch, 1)
            confidence = confidence.view(h_t.shape[0], 1, 1, 1)  # Expand for broadcasting
        else:
            confidence = torch.ones(h_t.shape[0], 1, 1, 1, device=h_t.device)
        
        # Step 2: WARP HIDDEN STATE (confidence-gated)
        if probs is not None and v_list is not None:
            if use_continuous_flow:
                h_t_warped = self.apply_flow_continuous(h_t, probs, v_list)
            else:
                h_t_warped = self.apply_flow_soft_efficient(h_t, probs, v_list)
            
            # Only warp if confident; otherwise keep original
            h_t = confidence * h_t_warped + (1 - confidence) * h_t
        # else: h_t unchanged (no velocity info)
        
        # Step 3: CONVOLVE WARPED HIDDEN STATE (clean, single conv)
        conv_h = self.conv_h(h_t)
        conv_h = self.norm_h(conv_h)
        
        # Step 4: ENCODE INPUT
        encoded_f = self.conv_u(f_t)
        encoded_f = self.norm_u(encoded_f)
        
        # Step 6: Combine with skip connection
        h_next = self.activation(conv_h + encoded_f)
        
        return h_next
        




class SeqtoSeqRNN(nn.Module):
    """
    Complete sequence-to-sequence model with discrete velocity prediction.
    """
    def __init__(self, input_channels, hidden_channels, height, width,
                    output_channels=None, h_kernel_size=3, u_kernel_size=3,
                    v_range=2, decoder_conv_layers=1,
                    use_attention=True, use_motion_diff=True, use_multiscale=True, use_velocity_gating=True):
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
            use_attention: Enable attention in velocity predictor
            use_motion_diff: Use motion difference features in velocity predictor
            use_multiscale: Use multi-scale processing in velocity predictor
            return_probs: If True, return velocity probabilities instead of velocities
        """
        super().__init__()

        self.height = height
        self.width = width
        self.output_channels = output_channels or input_channels
        self.v_range = v_range

        # Discrete velocity predictor 
        self.velocity_predictor = ParametricVelPrediction(input_channels=input_channels, 
                                                            v_range=v_range, 
                                                            hidden_dim=64,
                                                            use_attention=use_attention,
                                                            use_motion_diff=use_motion_diff,
                                                            use_multiscale=use_multiscale
                                                        )

        # Main RNN cell (uses integer shifts)
        self.cell = FERNN_Cell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size,
            use_velocity_gating=use_velocity_gating
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
                target_seq=None, return_probs=False, use_continuous_flow=False):
        """
        Forward pass for sequence prediction.

        Args:
            input_seq: (batch, T_in, C, H, W) - input sequence
            pred_len: int - number of frames to predict
            teacher_forcing_ratio: float - probability of using ground truth
            target_seq: (batch, pred_len, C, H, W) - ground truth for teacher forcing
            return_velocities: bool - whether to return velocity estimates

        Returns:
            predictions: (batch, pred_len, C, H, W) - predicted frames
            velocities: optional (batch, T_in+pred_len-1, num_v or 2) - velocity probabilities or velocities
        """
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device

        # Initialize hidden state
        h = self.init_hidden(batch, device)

        # Store velocities/probs if requested
        vel_probs = []

        # --- Encoder Phase: Process input sequence ---
        for t in range(T_in):
            f_t = input_seq[:, t]  # Current frame

            # Estimate velocity (need current and previous frame)
            if t == 0:
                # First frame: no previous frame, use zero velocity
                u_t = torch.zeros(batch, 2, device=device, dtype=torch.long)
                probs = torch.zeros(batch, self.velocity_predictor.num_v, device=device)
                probs[:, self.velocity_predictor.num_v // 2] = 1.0
            else:
                f_t_prev = input_seq[:, t-1]  # Previous frame
                u_t, _, probs = self.velocity_predictor(f_t, f_t_prev)

            # Update RNN (soft during training, hard during eval)
            h = self.cell(
                f_t,
                h,
                probs=probs,
                v_list=self.velocity_predictor.v_list,
                use_continuous_flow = use_continuous_flow
            )

            # Store velocity or probs
            if return_probs:
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

            u_t, _, probs = self.velocity_predictor(current_frame, f_t_prev)

            # Update RNN (soft during training, hard during eval)
            h = self.cell(
                current_frame,
                h,
                probs=probs,
                v_list=self.velocity_predictor.v_list,
                use_continuous_flow = use_continuous_flow
            )

            # Decode hidden state to frame prediction
            pred = self.decoder(h)
            predictions.append(pred)
            prev_frame = pred  # For next iteration

            # Store velocity or probs
            if return_probs:
                vel_probs.append(probs)

        # Stack predictions along time dimension
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_len, C, H, W)

        if return_probs:
            velocities_probs = torch.stack(vel_probs, dim=1)  # (batch, T_in+pred_len-1, num_v or 2)
            return predictions, velocities_probs
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
