import torch
import plotly.graph_objects as go


def plot_trajectory(o, d, colors=None, d_scale=1):
    # 2. Convert the PyTorch tensor to a NumPy array for plotting
    d = d_scale * d / ((d ** 2).sum(dim=-1).sqrt().unsqueeze(-1) + 1e-12) # Normalizes direction
    traj = o
    dirs = torch.stack([o, o + d, torch.zeros_like(o) * torch.nan], dim=-2).reshape(-1, 3) # start end none (none breaks trajectory)
    colors = colors.repeat_interleave(3, dim=-2) if colors is not None else None
    
    # 3. Create the interactive 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines',
        line=dict(
            color='black',
            width=3
        )
    ),go.Scatter3d(
        x=dirs[:, 0],
        y=dirs[:, 1],
        z=dirs[:, 2],
        mode='lines',
        line=dict(
            color=colors if colors is not None else 'red',
            width=3
        )
    )])
    
    # 4. Customize the layout
    fig.update_layout(
        title='Trajectory',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # 5. Display the plot (opens an interactive window)
    fig.show()
