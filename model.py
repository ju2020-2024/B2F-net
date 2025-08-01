import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, relu_rate=0.0, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation // 2,
                              dilation=dilation)
        self.relu = nn.LeakyReLU(negative_slope=relu_rate)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.downsample(x)

class TCNStack(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, num_layers=3, relu_rate=0.0, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_c = input_dim if i == 0 else hidden_dim
            layers.append(TemporalConvBlock(in_c, hidden_dim, kernel_size=3, dilation=dilation, relu_rate=relu_rate, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        out = self.net(x)
        return out.permute(0, 2, 1)  # (B, T, C)

class AttentionFusion(nn.Module):
    def __init__(self, feat_dim,relu_rate=0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(negative_slope=relu_rate),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x_cat = torch.cat([x1, x2], dim=-1)  # [B, T, 2D]
        attn_weights = self.attn(x_cat)      # [B, T, 1]
        return attn_weights * x1 + (1 - attn_weights) * x2  # [B, T, D]

class TemporalSpatialModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tcn_model_1 = TCNStack(input_dim=1024, hidden_dim=256, num_layers=3, relu_rate=args.relu_rate, dropout=args.dropout)
        self.tcn_model_2 = TCNStack(input_dim=1152, hidden_dim=256, num_layers=3, relu_rate=args.relu_rate, dropout=args.dropout)

        self.gru_model = nn.GRU(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)
        self.fusion = AttentionFusion(feat_dim=128,relu_rate=args.relu_rate)

    def forward(self, x, seq_len):
        xv = x[:, :, :1024]
        xa = x[:, :, 1024:]
        xv = self.tcn_model_1(xv)
        x1, _ = self.gru_model(xv)
        s1 = self.fc(x1)
        mil_s1 = self.mil_scores(s1, seq_len)
        x2_input = self.tcn_model_2(x)
        x2, _ = self.gru_model(x2_input)
        s2 = self.fc(x2)
        mil_s2 = self.mil_scores(s2, seq_len)
        x3 = self.fusion(x1, x2)
        s3 = self.fc(x3)
        mil_s3 = self.mil_scores(s3, seq_len)
        return {
            "frame_scores": [s1, s2, s3],
            "video_scores": [mil_s1, mil_s2, mil_s3]
        }

    def mil_scores(self, segment_scores, seq_len):
        segment_scores = segment_scores.squeeze(-1)  # (B, T)
        instance_scores = []
        for i in range(segment_scores.shape[0]):
            length = seq_len[i] if seq_len is not None else segment_scores.shape[1]
            k = max(1, int(length * 0.1))
            topk_vals, _ = torch.topk(segment_scores[i][:length], k=k, largest=True)
            weights = F.softmax(topk_vals, dim=0)
            tmp = torch.sum(topk_vals * weights).view(1)
            instance_scores.append(tmp)
        instance_scores = torch.cat(instance_scores, dim=0)
        return instance_scores
