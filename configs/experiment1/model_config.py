from torch import nn
from python.agent import CustomCNNFeatureExtractor

policy_name = "MlpPolicy"

hyperparams = dict(
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=0.0003,
    ent_coef=0.00,
    clip_range=0.2,
    n_epochs=10,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64],
            vf=[64]
        ),
        activation_fn=nn.Tanh
    )
)
