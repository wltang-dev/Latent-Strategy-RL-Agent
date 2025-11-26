Latent-Strategy-RL-Agent

Official implementation of a cognitive multi-agent decision-making system where
LLM-based agents evolve latent persuasion strategies through reflection, reinforcement signals, and long-term memory—all without any parameter fine-tuning.

This project demonstrates:

Multi-agent persuasion competition

Latent strategy learning (reflection → embedding → PCA → latent update)

Meta-controller with trust modeling

Cross-episode long-term memory using vector similarity

RL-augmented action selection in a dynamic grid world

Style-decoder learning with REINFORCE-like updates

Image + text multi-modal reasoning using OpenAI models

1. Core Idea

Each agent (emotion, rational, habit, risk_monitor, social_cognition):

Generates an action suggestion using LLM reasoning + a latent persuasion vector

Justifies its persuasion style, sampled through a learnable decoder

Receives RL rewards based on outcomes (goal, trap, social point, food, etc.)

Reflects on its failure/success, generating new embedding-based strategy vectors

Updates its persuasion latent via exponential moving average

Trains its style decoder using a small policy-gradient update

Stores reflection memory with semantic embeddings

Recalls similar past reflections using cosine similarity

Adjusts trust dynamically through shared outcomes

The meta-controller:

Reads agent outputs and trust levels

Incorporates retrieved cross-episode memory

Makes the final decision

Generates meta-reflection → produces meta state vectors

These vectors form episodic memory embeddings for long-term learning

This creates a dual-loop learning system:

Inner loop: agents persuade Meta within each step

Outer loop: reflections generate latent strategy evolution across steps/episodes

2. Key Components
1. Agent latent persuasion strategy

Each agent maintains:

persuasion_latent ∈ R^1536

Style decoder (W, b) for 7 persuasion aspects

Reflection & interaction memory

RL Q-table

Mood / stamina / career dynamics (emotion-driven)

2. Reflection-driven latent update

Each step generates:

Reflection text → embedding → PCA → latent strategy update


Latent update uses:

Reflection embedding

Reward

Whether the Meta accepted the suggestion

Global round ID

EMA update + normalization

3. Style Decoder (Policy Network)

For seven aspects (tone, honesty, emotion_use...), the system:

Maps latent vector → logits → softmax

Samples persuasion style

Converts styles into natural language instructions

Updates the decoder via:

ΔW, Δb ← reward × (onehot − probs) × latent_vector


(RL policy gradient)

4. Meta-Controller

Makes final decisions using:

Agent suggestions

Trust descriptors

Optional emotion override rule

Cross-episode memory bias

No map access → must rely on agents

5. Cross-Episode Memory System

Each episode generates:

Meta-reflection vectors

Compressed episode vector (mean of step vectors)

Future episodes retrieve similar memories through cosine similarity on map embeddings.

3. Environment

A multi-objective 10×10 cognitive grid world:

Agent

Goal

Food

Social targets

Traps

Event system triggers rewards for each agent differently.

4. Repository Structure
code-agent.py           # Main full implementation
emotion_memory.json     # Per-agent memory snapshots
strategy_evolution.jsonl# Latent strategy logs
experiment_results.json # Episode logs
figs/                   # Plots automatically generated

5.  How to Run
python code-agent.py


Make sure to add your OpenAI API key in:

client = OpenAI(api_key="YOUR_KEY")

Features Demonstrated for Employers / Researchers

This repository showcases:

✔ Multi-agent LLM architecture
✔ LLM-based RL with latent strategy evolution
✔ Embedding-based memory and PCA compression
✔ Multi-modal reasoning (text + environment images)
✔ Cognitive-inspired decision and reflection cycles
✔ Custom REINFORCE implementation
✔ Trust modeling and social cognition simulation
✔ Fully reproducible experiments

This project is suitable for:

MSc/PhD program portfolios

Applied AI researcher roles

RL / LLM reasoning engineer positions

Cognitive AI and agent foundation research teams

6.  License

MIT License.
