import gym
import numpy as np

UNLOCK_PREFIX = 'unlock_'


def compute_scores(percents):
    # Geometric mean with an offset of 1%.
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
    return scores


class CrafterStats(gym.Wrapper):

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        if done:
            achievements = []
            for achievement in info['achievements']:
                achievements.append(1 if info['achievements'][achievement] > 0.0 else 0)
                info['episode_extra_stats'][UNLOCK_PREFIX + achievement] = achievements[-1]

            info['episode_extra_stats']['done'] = True
            info['episode_extra_stats']['Score'] = compute_scores(np.array(achievements) * 100)
            info['episode_extra_stats']['Num_achievements'] = sum(achievements)
        return obs, reward, done, info
