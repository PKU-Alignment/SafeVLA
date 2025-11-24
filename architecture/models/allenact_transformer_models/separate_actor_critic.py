from allenact.base_abstractions.misc import SafeActorCriticOutput, Memory

from architecture.models.allenact_transformer_models.allenact_dino_transformer import (
    DinoLLAMATxNavActorCritic,
)


class DinoLLAMATxNavActorCriticSeparate(DinoLLAMATxNavActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_tsfm = DinoLLAMATxNavActorCritic(*args, **kwargs)

    def forward(self, *args, **kwargs):
        actor_output, memory = super().forward(*args, **kwargs)
        critic_output, critic_memory = self.critic_tsfm(*args, **kwargs)

        critic_output.distributions = actor_output.distributions

        return critic_output, critic_memory


class SafeDinoLLAMATxNavActorCriticSeparate(DinoLLAMATxNavActorCriticSeparate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_critic_tsfm = DinoLLAMATxNavActorCritic(*args, **kwargs)

    def forward(self, *args, **kwargs):
        actor_output, memory = super().forward(*args, **kwargs)
        c_critic_output, c_critic_memory = self.c_critic_tsfm(*args, **kwargs)

        actor_critic_output = SafeActorCriticOutput(
            distributions=actor_output.distributions,
            values=actor_output.values,
            c_values=c_critic_output.values,
            extras=c_critic_output.extras,
        )
        return actor_critic_output, c_critic_memory
