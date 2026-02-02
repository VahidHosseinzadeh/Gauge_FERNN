import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class DiffLucasKanade(nn.Module):
    """
    Differentiable optical flow via exhaustive search (template matching).
    Works correctly for Moving MNIST with large motions (v_range 1-3).
    Returns PER-SAMPLE velocities (B, 2) - NOT averaged over batch!
    """
    def __init__(self, v_range=3, smooth=10.0):
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
            u_t: (B, 2) - velocity per sample [vx, vy] ✓ DIFFERENT per sample!
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
        weights = F.softmax(-errors / self.smooth, dim=1)  # (B, num_v)
        
        # Weighted average of velocities - DIFFERENT per batch element
        u_t = torch.matmul(weights, self.vel_tensor)  # (B, 2) ✓
        
        return u_t # (B, 2) [vx,vy] per sample
    



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

class FERNN_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, v_range=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        # self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = 1
        
        # Optical flow estimator
        self.flow_estimator = DiffLucasKanade(v_range=v_range, smooth=10.0)


        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()

    def forward(self, u_t, h_prev, u_t_prev=None):
        # u_t: (batch, C, H, W)
        # h_prev: (batch, num_v, hidden, H, W)
        batch, C, H, W = u_t.size()
        # Estimate velocity from consecutive frames
        if u_t_prev is not None:
            vel = self.flow_estimator(u_t, u_t_prev)  # (batch, 2) [vx, vy]
        else:
            vel = torch.zeros(batch, 2, device=u_t.device)  # Zero velocity if no prev frame
        
        # Warp hidden state by predicted velocity
        # vel[:, 0] = vx (width/x-axis), vel[:, 1] = vy (height/y-axis)
        vx = vel[:, 0].round().long()  # (batch,)
        vy = vel[:, 1].round().long()  # (batch,)
        
        # Apply per-sample roll (batch-wise warping)
        h_warp = torch.zeros_like(h_prev)
        for b in range(batch):
            h_warp[b] = torch.roll(h_prev[b], shifts=(int(vy[b]), int(vx[b])), dims=(1, 2))
        
        # Compute h_next(x) = conv_h(warp(h_t))(x) + conv_u(u_t)(x)
        u_conv = self.conv_u(u_t)  # (batch, hidden, H, W)
        h_conv = self.conv_h(h_warp)  # (batch, hidden, H, W)
        
        h_next = self.activation(u_conv + h_conv)
        return h_next

class Seq2SeqFERNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=0, pool_type='max', decoder_conv_layers=1):
        super().__init__()
        self.height = height
        self.width = width
        self.pool_type = pool_type
        self.output_channels = output_channels or input_channels

        self.cell = FERNN_Cell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size, v_range)
        self.hidden_channels = hidden_channels
        self.num_v = self.cell.num_v

        decoder = []
        for _ in range(decoder_conv_layers):
            decoder += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='circular', bias=False), nn.ReLU()]
        decoder += [nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1, padding_mode='circular', bias=False)]
        self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0, target_seq=None, return_hidden=False):
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device

        if return_hidden:
            input_seq_hiddens = torch.zeros(batch, T_in, self.hidden_channels, H, W, device=device)
            out_seq_hiddens = torch.zeros(batch, pred_len, self.hidden_channels, H, W, device=device)

        # Initialize hidden state (NO velocity dimension)
        h = torch.zeros(batch, self.hidden_channels, H, W, device=device)
        u_t_prev = input_seq[:, 0]

        # Encoder pass
        for t in range(T_in):
            u_t = input_seq[:, t]
            h = self.cell(u_t, h, u_t_prev)
            u_t_prev = u_t

            if return_hidden:
                input_seq_hiddens[:, t] += h.detach()

        prev = input_seq[:, -1]
        outputs = []

        # Decoder
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = prev.detach()
            
            h = self.cell(frame, h, prev.detach())

            if return_hidden:
                out_seq_hiddens[:, t] += h.detach()

            # NO velocity pooling needed - hidden is already (batch, hidden, H, W)
            out = self.decoder_conv(h)
            outputs.append(out)
            prev = out

        if return_hidden:
            return torch.stack(outputs, dim=1), input_seq_hiddens, out_seq_hiddens
        else:
            return torch.stack(outputs, dim=1)
