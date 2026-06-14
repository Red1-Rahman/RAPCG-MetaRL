\begin{table}[htbp]
\caption{Hardware Profile Comparison: Training/Inference Platform vs. Cross-Platform Inference}
\label{tab:hardware_profiles}
\centering
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Component} & \textbf{Intel i5-13500 + RTX 3060 Ti} & \textbf{AMD Ryzen 5 3550H (Infer Only)} \\
\midrule
CPU & Intel i5-13500 (14c/20t, 2.5 GHz) & AMD Ryzen 5 3550H (8c/16t, 3.7 GHz) \\
GPU/APU & NVIDIA RTX 3060 Ti (8GB GDDR6) & AMD RX 560x / Vega 8 iGPU \\
RAM & 16GB DDR4-3600 & 8GB DDR4-2667 SODIMM \\
Use Case & Training + Inference & Inference Only (CPU-based) \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{RAPCG-MetaRL Component Implementation Status}
\label{tab:impl_status}
\centering
\small
\begin{tabular}{@{}llc@{}}
\toprule
\textbf{Component} & \textbf{Status} & \textbf{Section} \\
\midrule
PPO/A2C Training Pipeline & Implemented & \ref{sec:experiments} \\
Resource-Aware Reward Shaping & Implemented & \ref{sec:method} \\
Hardware Telemetry (\texttt{psutil}/\texttt{pynvml}) & Implemented & \ref{sec:method} \\
Adaptive Batch Scheduling & Proposed & \ref{sec:method} \\
MAML Meta-RL Controller & Proposed & \ref{sec:method} \\
Hybrid PCG Ensemble & Proposed & \ref{sec:method} \\
Unity/Unreal Integration & Proposed & \ref{sec:implementation} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{CPU-Based Training Run Metrics (3,072 steps)}
\label{tab:preliminary}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Training Steps & 3,072 \\
Mean Episode Reward & 5.2 \\
FPS (Training) & 2.9 \\
CPU Utilization (avg) & 29\% \\
RAM Usage (avg) & 12.5 GB \\
GPU Utilization (avg) & <1\% \\
GPU Memory Usage & Minimal \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Zelda PPO CUDA Training Run Metrics (20,096 steps)}
\label{tab:gpu_training}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Training Steps & 20,096 \\
Total Episodes & 566 \\
Training Duration & 113.2 minutes \\
Mean Episode Reward & 5.60 \\
Episode Reward Range & $-39.0$ to $+53.0$ \\
Learning Improvement (early $\to$ late) & $-8.54 \to +11.84$ \\
CPU Utilization (avg) & 3.68\% \\
RAM Usage (avg) & 48.15\% (7.60\,GB) \\
GPU Utilization (avg) & 0.86\% \\
GPU Memory Usage (avg) & 824.4\,MB (10.06\% VRAM) \\
GPU Memory Usage (peak) & 938.2\,MB (11.45\% VRAM) \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Zelda PPO CUDA Inference Metrics (20 levels)}
\label{tab:zelda_inference}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Levels Generated & 20 \\
Mean Generation Time & 5,176.1\,ms \\
Min / Max Generation Time & 2,852.0\,ms / 11,004.6\,ms \\
Mean Inference per Step & 2.64\,ms \\
Min / Max Inference per Step & 1.68\,ms / 5.06\,ms \\
Mean Steps per Level & 45.6 \\
Min / Max Steps per Level & 25 / 98 \\
Mean Total Reward & 14.10 \\
Min / Max Total Reward & $-30.0$ / $+53.0$ \\
Mean Diversity & 0.0675 \\
Mean Complexity & 0.8300 \\
GPU VRAM During Inference & 7.2–8.3\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Sokoban PPO CUDA Training Run Metrics (20,096 steps)}
\label{tab:sokoban_gpu_training}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Training Steps & 20,096 \\
Total Episodes & 2,018 \\
Training Duration & 112.6 minutes \\
Mean Episode Reward & 5.39 \\
Episode Reward Range & $-19.0$ to $+34.0$ \\
Learning Improvement (early $\to$ late) & $+2.84 \to +6.66$ \\
CPU Utilization (avg) & 3.81\% \\
RAM Usage (avg) & 49.37\% (7.79\,GB) \\
GPU Utilization (avg) & 0.84\% \\
GPU Memory Usage (avg) & 815.9\,MB (9.96\% VRAM) \\
GPU Memory Usage (peak) & 880.9\,MB (10.75\% VRAM) \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Sokoban PPO CUDA Inference Metrics (20 levels)}
\label{tab:sokoban_inference}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Levels Generated & 20 \\
Mean Generation Time & 1,586.3\,ms \\
Min / Max Generation Time & 653.6\,ms / 2,533.0\,ms \\
Mean Inference per Step & 3.19\,ms \\
Min / Max Inference per Step & 1.53\,ms / 15.15\,ms \\
Mean Steps per Level & 13.4 \\
Min / Max Steps per Level & 5 / 22 \\
Mean Total Reward & 8.25 \\
Min / Max Total Reward & $-4.0$ / $+28.0$ \\
Mean Diversity & 0.1540 \\
Mean Complexity & 0.9667 \\
GPU VRAM During Inference & 10.3–10.6\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Zelda PPO CPU Inference on AMD Ryzen 5 3550H (20 levels)}
\label{tab:zelda_amd_inference}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Levels Generated & 20 \\
Mean Generation Time & 4,933.2\,ms \\
Std Dev Generation Time & 1,258.2\,ms \\
Min / Max Generation Time & 2,761.9\,ms / 7,062.1\,ms \\
Mean Inference per Step & 1.01\,ms \\
Std Dev Inference per Step & 0.04\,ms \\
Min / Max Inference per Step & 0.97\,ms / 1.09\,ms \\
Mean Steps per Level & 47.1 \\
Min / Max Steps per Level & 26 / 68 \\
Mean Total Reward & 15.10 \\
Min / Max Total Reward & $0.0$ / $+37.0$ \\
Mean Diversity & 0.0760 \\
Mean Complexity & 0.8800 \\
Mean CPU Usage & 7.0\% \\
RAM Usage Change & 0.07\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Sokoban PPO CPU Inference on AMD Ryzen 5 3550H (20 levels)}
\label{tab:sokoban_amd_inference}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Levels Generated & 20 \\
Mean Generation Time & 1,442.9\,ms \\
Std Dev Generation Time & 793.4\,ms \\
Min / Max Generation Time & 653.6\,ms / 4,310.7\,ms \\
Mean Inference per Step & 1.15\,ms \\
Std Dev Inference per Step & 0.64\,ms \\
Min / Max Inference per Step & 0.94\,ms / 3.86\,ms \\
Mean Steps per Level & 13.1 \\
Min / Max Steps per Level & 6 / 41 \\
Mean Total Reward & 5.95 \\
Min / Max Total Reward & $-3.0$ / $+15.0$ \\
Mean Diversity & 0.1440 \\
Mean Complexity & 0.9667 \\
Mean CPU Usage & 2.5\% \\
RAM Usage Change & 0.00\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\caption{Comprehensive Algorithm Comparison: A2C vs PPO across Domains}
\label{tab:algorithm_comparison}
\centering
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
\textbf{Algorithm} & \textbf{Domain} & \textbf{Episodes} & \textbf{Duration} & \textbf{Mean Ep.~Reward} & \textbf{Early $\to$ Late} & \textbf{CPU\%} & \textbf{RAM\%} \\
\midrule
A2C & Zelda & 492 & 128.2 min & $-8.92$ & $-14.6 \to -3.0$ & 4.7 & 58.8 \\
A2C & Sokoban & 1,291 & 56.3 min & 0.89 & $-1.9 \to +3.2$ & 4.4 & 63.2 \\
\midrule
PPO & Zelda & 566 & 113.2 min & 5.60 & $-8.54 \to +11.84$ & 3.68 & 48.15 \\
PPO & Sokoban & 2,018 & 112.6 min & 5.39 & $+2.84 \to +6.66$ & 3.81 & 49.37 \\
\bottomrule
\end{tabular}
\end{table}
