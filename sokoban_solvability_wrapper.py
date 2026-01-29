"""
Sokoban Solvability Wrapper
Adds strong penalties for unsolvable levels to prevent deadlocks.
"""
import gym
import numpy as np


class SokobanSolvabilityWrapper(gym.Wrapper):
    """
    Wrapper that STRONGLY enforces solvability for Sokoban levels.
    
    Unsolvable Sokoban levels are completely pointless - they can never be won.
    This wrapper:
    1. Applies MASSIVE penalties for unsolvable levels
    2. Optionally terminates episodes on unsolvable levels (forces re-generation)
    3. Tracks how often the agent generates garbage levels
    """
    
    def __init__(self, env, unsolvable_penalty=10.0, min_solution_length=5, 
                 max_solution_length=50, terminate_on_unsolvable=False):
        """
        Args:
            env: Base Sokoban PCGRL environment
            unsolvable_penalty: Penalty for unsolvable levels (higher = stronger)
            min_solution_length: Minimum acceptable solution length
            max_solution_length: Maximum acceptable solution length
            terminate_on_unsolvable: If True, end episode when unsolvable level created
        """
        super().__init__(env)
        self.unsolvable_penalty = unsolvable_penalty
        self.min_solution_length = min_solution_length
        self.max_solution_length = max_solution_length
        self.terminate_on_unsolvable = terminate_on_unsolvable
        
        # Statistics
        self.total_levels = 0
        self.unsolvable_levels = 0
        self.solvable_levels = 0
        self.solution_lengths = []
        
    def reset(self, **kwargs):
        """Reset the environment."""
        obs = self.env.reset(**kwargs)
        
        # Reset statistics
        self.total_levels = 0
        self.unsolvable_levels = 0
        self.solvable_levels = 0
        self.solution_lengths = []
        
        return obs
    
    def step(self, action):
        """
        Step with solvability checking.
        """
        obs, reward, done, info = self.env.step(action)
        
        # Extract stats from the environment
        # gym-pcgrl stores stats in info dictionary
        if 'stats' in info:
            stats = info['stats']
        else:
            # If not in info, try to get from environment directly
            try:
                stats = self.env.unwrapped._prob.get_stats(
                    self.env.unwrapped._rep._map
                )
            except:
                # If can't get stats, return unmodified
                return obs, reward, done, info
        
        # Check if level is complete (has player, crates match targets, single region)
        is_complete = (
            stats.get('player', 0) == 1 and
            stats.get('crate', 0) > 0 and
            stats.get('crate', 0) == stats.get('target', 0) and
            stats.get('regions', 0) == 1
        )
        
        if is_complete:
            self.total_levels += 1
            
            # Check if solvable
            solution = stats.get('solution', [])
            dist_win = stats.get('dist-win', float('inf'))
            
            is_solvable = len(solution) > 0 and dist_win == 0
            
            if is_solvable:
                self.solvable_levels += 1
                self.solution_lengths.append(len(solution))
                
                # BIG reward for solvable levels - this is what we want!
                solvability_reward = 10.0
                
                # Extra bonus for good solution length
                if self.min_solution_length <= len(solution) <= self.max_solution_length:
                    solvability_reward += 5.0
                
                reward += solvability_reward
                
                if not isinstance(info, dict):
                    info = {}
                info['solvable'] = True
                info['solution_length'] = len(solution)
                info['solvability_reward'] = solvability_reward
                
            else:
                # MASSIVE PENALTY - unsolvable levels are WORTHLESS
                self.unsolvable_levels += 1
                
                # Apply crushing penalty
                penalty = self.unsolvable_penalty
                reward -= penalty
                
                # Optionally terminate episode - force agent to start over
                if self.terminate_on_unsolvable:
                    done = True
                
                if not isinstance(info, dict):
                    info = {}
                info['solvable'] = False
                info['solution_length'] = 0
                info['solvability_penalty'] = penalty
                info['dist_win'] = dist_win
                info['terminated_unsolvable'] = self.terminate_on_unsolvable
        
        # Add statistics to info
        if self.total_levels > 0:
            info['solvability_rate'] = (
                self.solvable_levels / self.total_levels
            )
            info['unsolvable_rate'] = (
                self.unsolvable_levels / self.total_levels
            )
            if len(self.solution_lengths) > 0:
                info['avg_solution_length'] = np.mean(self.solution_lengths)
        
        return obs, reward, done, info
    
    def get_statistics(self):
        """Get solvability statistics."""
        if self.total_levels == 0:
            return {
                'total_levels': 0,
                'solvable_levels': 0,
                'unsolvable_levels': 0,
                'solvability_rate': 0.0,
                'avg_solution_length': 0.0
            }
        
        return {
            'total_levels': self.total_levels,
            'solvable_levels': self.solvable_levels,
            'unsolvable_levels': self.unsolvable_levels,
            'solvability_rate': self.solvable_levels / self.total_levels,
            'avg_solution_length': (
                np.mean(self.solution_lengths) 
                if len(self.solution_lengths) > 0 else 0.0
            )
        }
