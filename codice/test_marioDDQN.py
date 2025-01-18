import os
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from pathlib import Path
import time
import datetime
import matplotlib.pyplot as plt
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from collections import deque
import random
import sys
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage


# Wrappers
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
            obs, reward, done, info = self.env.step(action)

            # Favorire avanzamento
            reward += (info.get("x_pos", 0) / 5)

            # Penalità per caduta o morte
            if done and info.get("life", 2) < 2:
                reward -= 500

            # Bonus per raggiungere il traguardo
            if "flag_get" in info and info["flag_get"]:
                reward += 1000

            return obs, reward, done, info


class ReplayBuffer:
    def __init__(self, size=500000, state_dim=(4, 84, 84)):
        self.max_size = size  # Capacità massima del buffer
        self.state_dim = state_dim

        # Inizializza gli array per il buffer
        self.states = torch.zeros((size, *state_dim), dtype=torch.float32)
        self.next_states = torch.zeros((size, *state_dim), dtype=torch.float32)
        self.actions = torch.zeros(size, dtype=torch.long)
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.dones = torch.zeros(size, dtype=torch.float32)

        self.buffer_index = 0
        self.buffer_filled = 0

    def add(self, state, next_state, action, reward, done):
        # Converti `done` in tipo float
        done = float(done)

        # Aggiungi i dati al buffer circolare
        self.states[self.buffer_index] = state
        self.next_states[self.buffer_index] = next_state
        self.actions[self.buffer_index] = action
        self.rewards[self.buffer_index] = reward
        self.dones[self.buffer_index] = done

        # Aggiorna l'indice e il contatore
        self.buffer_index = (self.buffer_index + 1) % self.max_size
        self.buffer_filled = min(self.buffer_filled + 1, self.max_size)


    def sample(self, batch_size):
        # Estrai un campione casuale dal buffer
        indices = np.random.choice(self.buffer_filled, batch_size, replace=False)

        states = self.states[indices]
        next_states = self.next_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        return states, next_states, actions, rewards, dones


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape, antialias=True), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation

# Neural Network
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"Expecting input size (84, 84), got ({h}, {w})")
        self.online = self.__build_cnn(c, output_dim)
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, use_online: bool):
        if use_online:
            return self.online(input)
        else:
            return self.target(input)


    def __build_cnn(self, c, output_dim):
        model = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Testa con un input fittizio
        test_input = torch.zeros(1, c, 84, 84)
        output_size = model(test_input).shape[1]
        return nn.Sequential(
            model,
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )


# Mario Agent
class Mario:
    def __init__(self, state_dim, action_dim, save_dir, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = torch.jit.script(MarioNet(self.state_dim, self.action_dim)).float().to(self.device)
        self.memory = ReplayBuffer(size=100000)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)  # Riduci il learning rate
        self.loss_fn = nn.SmoothL1Loss()
        self.exploration_rate = 1.0  # Inizia con un'esplorazione più bassa
        self.exploration_rate_decay = 0.999995  # Decay ancora più lento
        self.exploration_rate_min = 0.01  # Permetti un po' di esplorazione anche nelle fasi avanzate
        self.curr_step = 0
        self.gamma = 0.99
        self.burnin = 2e4  # Raccogli più esperienze iniziali (20.000)
        self.learn_every = 5  # Aggiorna il modello ogni 5 step
        self.sync_every = 10000  # Sincronizza il modello target ogni 10.000 step
        self.save_every = 5e4
        self.batch_size = batch_size  # Aggiungi la dimensione del batch


    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state.__array__(), device=self.device).unsqueeze(0)
            action_values = self.net(state, use_online=True)
            action = torch.argmax(action_values, axis=1).item()

        self.curr_step += 1  # Incrementa il contatore degli step
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate * self.exploration_rate_decay
        )
        return action


    def cache(self, state, next_state, action, reward, done):
        state = torch.tensor(state.__array__(), dtype=torch.float32)
        next_state = torch.tensor(next_state.__array__(), dtype=torch.float32)
        self.memory.add(state, next_state, action, reward, done)

    def recall(self):
        states, next_states, actions, rewards, dones = self.memory.sample(self.batch_size)
        return (
            states.to(self.device),
            next_states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
        )


    def learn(self):
        # Sincronizza il modello target
        if self.curr_step > 0 and self.curr_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        # Salva il modello
        if self.curr_step > 0 and self.curr_step % self.save_every == 0:
            checkpoint_file = self.save_dir / f"mario_{self.curr_step}.chkpt"
            torch.save(self.net.state_dict(), checkpoint_file)
            print(f"Saved model at {checkpoint_file}")

        # Non imparare prima di burn-in o ogni passo non multiplo di learn_every
        if self.curr_step < self.burnin or self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.net(state, use_online=True)[torch.arange(self.batch_size), action]
        best_action = torch.argmax(self.net(next_state, use_online=True), dim=1)

        td_target = reward + self.gamma * (1 - done.float()) * self.net(next_state, use_online=False)[torch.arange(self.batch_size), best_action]

        loss = self.loss_fn(td_est, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10)  # Clipping del gradiente
        self.optimizer.step()

        return loss.item(), td_est.mean().item()


    def save(self, step):
        save_path = self.save_dir / f"mario_{step}.pt"
        torch.jit.save(self.net, save_path)  # Salva il modello come ScriptModule
        print(f"Saved optimized model at {save_path}")


    def load(self, checkpoint_path):
        self.net = torch.jit.load(checkpoint_path, map_location=self.device)  # Carica il modello come ScriptModule
        print(f"Loaded optimized model from {checkpoint_path}")


# Logger
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon:.6f} - "
            f"Mean Reward {mean_ep_reward} - Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:.6f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

# Main function
def main():
    # Inizializza ambiente e modello
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=2)
    env = RewardShaping(env)  # Applica il nuovo wrapper qui
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(save_dir)

    print("Seleziona una modalità:")
    print("1: Nuovo addestramento")
    print("2: Continua addestramento")
    print("3: Gioca con il modello addestrato")
    choice = input("Inserisci 1, 2 o 3: ")

    mario = None  # Inizializzazione del modello

    if choice == "1":
        # Nuovo addestramento
        mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    elif choice == "2":
        # Continua addestramento
        checkpoint_path = input("Inserisci il percorso del checkpoint salvato: ")
        if os.path.exists(checkpoint_path):
            mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
            mario.net.load_state_dict(torch.load(checkpoint_path))
            print(f"Modello caricato da {checkpoint_path}")
        else:
            print("Checkpoint non trovato. Esco...")
            return
    elif choice == "3":
        # Gioca con il modello addestrato
        checkpoint_path = input("Inserisci il percorso del modello addestrato: ")
        if os.path.exists(checkpoint_path):
            mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
            mario.net.load_state_dict(torch.load(checkpoint_path))
            print(f"Modello caricato da {checkpoint_path}")
            play(env, mario)
        else:
            print("Modello non trovato. Esco...")
            return
    else:
        print("Scelta non valida. Esco...")
        return

    # Addestramento
    try:
        episodes = 5000
        render = True  # Cambia a False se non vuoi il rendering per ogni episodio

        for e in range(episodes):
            state = env.reset()
            episode_reward = 0  # Per tenere traccia della ricompensa totale dell'episodio
            won = False  # Flag per determinare se ha vinto l'episodio

            while True:
                if render:
                    env.render()

                action = mario.act(state)
                next_state, reward, done, info = env.step(action)

                # Cache the experience e apprendimento
                mario.cache(state, next_state, action, reward, done)
                loss, mean_q = mario.learn()
                logger.log_step(reward, loss, mean_q)

                # Accumula ricompensa
                episode_reward += reward

                # Controlla se Mario ha vinto
                if "flag_get" in info and info["flag_get"]:
                    won = True

                # Aggiorna lo stato corrente
                state = next_state

                # Termina l'episodio se è finito
                if done:
                    break

            # Log dell'episodio
            logger.log_episode()

            # Stampa messaggio di vittoria
            if won:
                checkpoint_file = save_dir / f"mario_win_{e}.chkpt"
                mario.save(e)
                print(f"🎉 Mario ha vinto l'episodio {e}! Ricompensa totale: {episode_reward}")

            # Registra le metriche ogni 20 episodi o all'ultimo episodio
            if e % 20 == 0 or e == episodes - 1:
                logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


    except KeyboardInterrupt:
        # Salva il modello in caso di interruzione
        print("Interruzione manuale! Salvataggio del modello...")
        checkpoint_file = save_dir / f"mario_interrupt_{mario.curr_step}.chkpt"
        torch.save(mario.net.state_dict(), checkpoint_file)
        print(f"Modello salvato in {checkpoint_file}")
    finally:
        env.close()


def play(env, mario):
    """Modalità gioco con il modello addestrato."""
    try:
        state = env.reset()
        while True:
            env.render()
            action = mario.act(state)
            state, _, done, _ = env.step(action)
            if done:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()