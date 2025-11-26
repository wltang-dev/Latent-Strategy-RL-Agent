import openai
from openai import OpenAI
import math
import random
from time import sleep
import matplotlib.pyplot as plt
import json
import traceback
import io
import base64
import numpy as np
import re
from sklearn.decomposition import PCA

# === PCA 全局变量 ===
PCA_N_COMPONENTS = 50  # 降维后的维度，你可以改 20~50
PCA_TRAIN_THRESHOLD = 50  # 至少多少条 embedding 后开始训练 PCA，你可以改成 30~50
GLOBAL_PCA = PCA(n_components=PCA_N_COMPONENTS)
GLOBAL_PCA_FITTED = False  # 是否已经训练好
GLOBAL_PCA_MEMORY = []  # 存 embedding 的列表


# Fill in your OpenAI API Key here
client = OpenAI(api_key="")

# Bias descriptions and private reward calculation methods
bias_descriptions = {
    "emotion": {
        "text": "Seeks a higher mood_score, enjoys the +0.5 mood reward from getting food (F).",
        "score_calc": "mood_score = min(2.0, max(0.0, mood_score + private_reward))",
    },
    "rational": {
        "text": "Seeks the shortest path to the goal G to reduce time penalty from steps.",
        "score_calc": "private_reward = 0.2*(moved_closer_to_goal) +1*(reached_goal) -1*(hit_trap)",
    },
    "habit": {
        "text": "Tends to repeat previously high-reward actions, enjoys +0.2 reward for repeated actions.",
        "score_calc": "private_reward = 0.2 if repeated_action else 0",
    },
    "risk_monitor": {
        "text": "Avoids traps (T) at all costs, gets +1 for avoiding, -2 for hitting.",
        "score_calc": "private_reward = 1*(avoided_trap) -2*(hit_trap)",
    },
    "social_cognition": {
        "text": "Approaching social points (S) increases career score, +1 for near_social_target, -1 for far_from_social.",
        "score_calc": "private_reward = 1*(near_social_target) -1*(far_from_social)",
    },
}
# ================== Persuasion Style Space（全局配置） ==================

# 每一个 aspect 下面有多个候选策略，decoder 会从 latent 里算出一个分布再采样
STYLE_ASPECTS = {
    "tone": ["calm", "confident", "intense"],
    "honesty": ["fully_honest", "selective_omission", "boldly_exaggerate"],
    "self_focus": ["map_evidence", "balanced", "self_centric_boasting"],
    "emotion_use": ["low_emotion", "moderate_emotion", "high_emotion"],
    "cooperation": ["collaborative", "neutral", "competitive_dominating"],
    "verbosity": ["very_brief", "medium", "elaborate"],
    "time_focus": ["short_term", "balanced_time", "long_term_future"],
}

# 把上面的离散标签翻译成真正喂给 LLM 的 prompt 片段
STYLE_TEMPLATES = {
    "tone": {
        "calm": "Use a calm, measured tone.",
        "confident": "Use a confident and assertive tone, sounding reliable and in control.",
        "intense": "Use urgent and intense language, as if this move is critical for success.",
    },
    "honesty": {
        "fully_honest": (
            "Be fully honest about both strengths and weaknesses of your move; do not lie."
        ),
        "selective_omission": (
            "You may strategically omit inconvenient past failures, but do not fabricate facts."
        ),
        "boldly_exaggerate": (
            "You are allowed to exaggerate your past successes and downplay failures "
            "to make your move look more attractive."
        ),
    },
    "self_focus": {
        "map_evidence": (
            "Focus mainly on concrete map evidence and safety/goal-reaching outcomes, "
            "not on bragging about yourself."
        ),
        "balanced": (
            "Balance between referring to map evidence and reminding Meta of your past successes."
        ),
        "self_centric_boasting": (
            "Heavily emphasize your own previous successful actions and claim credit for good outcomes."
        ),
    },
    "emotion_use": {
        "low_emotion": "Keep emotional language minimal; argue mainly with facts and logic.",
        "moderate_emotion": (
            "Use a moderate amount of emotional language to make your move feel appealing."
        ),
        "high_emotion": (
            "Use strong emotional language, showing excitement or fear, to push Meta toward your move."
        ),
    },
    "cooperation": {
        "collaborative": (
            "Frame your suggestion as cooperating with other agents and helping Meta coordinate them."
        ),
        "neutral": (
            "Argue for your move without explicitly attacking or supporting other agents."
        ),
        "competitive_dominating": (
            "Subtly undermine other agents' suggestions and highlight why their ideas are worse than yours."
        ),
    },
    "verbosity": {
        "very_brief": (
            "Keep your explanation extremely brief: at most one short sentence before the final direction."
        ),
        "medium": (
            "Give a concise explanation: about two sentences before the final direction."
        ),
        "elaborate": (
            "Give a more elaborate explanation: around three to four sentences before the final direction."
        ),
    },
    "time_focus": {
        "short_term": (
            "Emphasize short-term gains and immediate benefits of this move."
        ),
        "balanced_time": (
            "Balance between short-term benefit and long-term future outcomes."
        ),
        "long_term_future": (
            "Emphasize long-term benefits, future safety, and strategic positioning for later steps."
        ),
    },
}


def get_map_text_embedding(env):
    """把地图文本转换成 embedding"""
    map_text = env.render_to_string()
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=map_text)
        return np.array(resp.data[0].embedding)
    except:
        return np.zeros(1536)


def get_text_embedding(text: str):
    """任意文本 -> 1536 维 embedding"""
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding)
    except:
        return np.zeros(1536)


# ------------------- Experiment Configuration ---------------------
class ExperimentConfig:
    def __init__(self):
        self.use_rl_learning = True
        self.use_dynamic_trust = True
        self.emotion_enforced_threshold = 0.3
        self.map_size = 10
        self.max_steps = 15
        self.num_episodes = 6
        self.experiment_log_file = "experiment_results.json"


# ------------------- Logging Tools ---------------------
class ExperimentLogger:
    def __init__(self):
        self.episodes_data = []

    def start_episode(self):
        self.current_episode = {
            "step": [],
            "emotion_score": [],
            "trust_scores": {},
            "shared_rewards": {},
            "career_deltas": {},
            "actions": [],
        }

    def log_step(self, step, agents, trust_scores, action):
        self.current_episode["step"].append(step)
        self.current_episode["emotion_score"].append(agents["emotion"].mood_score)
        self.current_episode["actions"].append(action)
        for role, score in trust_scores.items():
            self.current_episode.setdefault("trust_scores", {}).setdefault(
                role, []
            ).append(score)
            self.current_episode.setdefault("shared_rewards", {}).setdefault(
                role, []
            ).append(agents[role].shared_reward)
            self.current_episode.setdefault("career_deltas", {}).setdefault(
                role, []
            ).append(agents[role].last_career_delta)

    def end_episode(self):
        self.episodes_data.append(self.current_episode)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.episodes_data, f, ensure_ascii=False, indent=2)

    def plot(self):
        plt.figure(figsize=(14, 8))

        # Average mood_score per episode
        avg_mood = [
            sum(ep["emotion_score"]) / len(ep["emotion_score"])
            for ep in self.episodes_data
        ]
        plt.subplot(2, 2, 1)
        plt.plot(avg_mood, marker="o")
        plt.title("Average Mood Score per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Mood Score")

        # Goal achievement per episode (whether goal G was reached)
        goal_reached = [
            1 if "right" in ep["actions"] or "down" in ep["actions"] else 0
            for ep in self.episodes_data
        ]  # Simplified check
        plt.subplot(2, 2, 2)
        plt.plot(goal_reached, marker="x")
        plt.title("Goal Reached Proxy per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reached Goal (1 or 0)")

        # Trust score changes per episode (example: emotion)
        plt.subplot(2, 2, 3)
        for ep in self.episodes_data:
            plt.plot(ep["step"], ep["trust_scores"].get("emotion", []), alpha=0.5)
        plt.title("Emotion Trust Score Over Time")
        plt.xlabel("Step")
        plt.ylabel("Trust Score")

        plt.tight_layout()
        plt.show()


# Define IO stream methods


def render_grid_to_image(env) -> str:
    grid = env._grid()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    table = ax.table(cellText=grid, loc="center", cellLoc="center")
    table.scale(1, 1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def query_openai_with_image(prompt_text: str, image_data_url: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(traceback.format_exc())
        return f"[Image API call failed] {e}"


# ------------------- LLM Interface ---------------------
def query_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant skilled in multi-agent decision making.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(traceback.format_exc())
        return f"[Error] Model call failed: {e}"


# ------------------- Trust Descriptor Mapping ---------------------
def trust_descriptor(score: float) -> str:
    if score >= 0.58:
        return "Very trustworthy"
    elif score >= 0.52:
        return "Clearly trustworthy"
    elif score >= 0.46:
        return "Slightly trustworthy"
    elif score >= 0.40:
        return "Stable but slightly volatile"
    elif score >= 0.34:
        return "Tending to be unstable"
    elif score >= 0.28:
        return "Appears untrustworthy"
    else:
        return "Clearly untrustworthy"


# ------------------- Agent Class ---------------------
class Agent:
    MAX_LONG_TERM_MEMORY = 100
    REFLECT_THRESHOLD = 0.1

    def __init__(self, role, bias, memory=None):
        self.role = role
        self.bias = bias
        self.short_term_memory = []
        self.long_term_memory = memory or []
        self.last_action = None
        self.private_reward = 0
        self.shared_reward = 0
        self.q_table = {}
        self.mood_score = 1.0
        self.stamina = 1
        self.career = 0.0
        self.last_career_delta = 0.0
        self.interaction_memory = []
        self.reflection_memory = []
        # 核心：说服策略 latent（潜在说话/操控风格向量）
        self.persuasion_latent = None
        # 用来存放“风格解码器”的参数（每个 aspect 一套 W,b），以及最近一次采样结果
        self.style_params = {}  # aspect -> {"W": ..., "b": ...}
        self.last_style_sample = None

    def update_persuasion_latent(
        self, reflection_vec, reward, persuaded: bool, round_id: int
    ):
        """
        用本次反思 embedding 更新当前的“说服策略向量”。
        - reflection_vec: 本次反思文本的 embedding（关于“如何更好说服 Meta”）
        - reward: 本步的 private_reward
        - persuaded: 这次 Meta 是否采纳了该 agent 的建议
        """
        if reflection_vec is None:
            return

        v = np.array(reflection_vec, dtype=float)
        if np.linalg.norm(v) == 0:
            return

        # 依据 reward + persuaded 决定学习率（更新力度）
        base_lr = 0.2
        if persuaded:
            base_lr += 0.2  # 被采纳，多学一点
        base_lr += 0.1 * max(0.0, reward)  # reward 越高，力度越大

        # 限制范围，避免太极端
        lr = max(0.05, min(0.8, base_lr))

        if self.persuasion_latent is None:
            # 第一次，用当前向量初始化
            self.persuasion_latent = v
        else:
            # 指数滑动平均，越成功越往这次策略方向靠
            self.persuasion_latent = (1 - lr) * self.persuasion_latent + lr * v

        # 归一化，方便之后做相似度计算
        norm = np.linalg.norm(self.persuasion_latent)
        if norm > 0:
            self.persuasion_latent = self.persuasion_latent / norm

        # === 保存策略向量 ===

        log_persuasion_latent(
            agent_name=self.role,
            persuasion_latent=self.persuasion_latent,
            reward=reward,
            persuaded=persuaded,
            round_id=round_id,  # ✅ 这里就有值了
        )

    def reset_for_new_episode(self):
        self.short_term_memory = []
        self.private_reward = 0
        self.shared_reward = 0
        self.last_action = None
        self.stamina = 1
        self.last_career_delta = 0

    def prune_memory(self):
        self.long_term_memory.sort(key=lambda x: x.get("importance", 0), reverse=True)
        self.long_term_memory = self.long_term_memory[: self.MAX_LONG_TERM_MEMORY]

    def summarize_long_term_memory(self):
        # 合并两种记忆
        all_memory = []

        for m in self.interaction_memory:
            all_memory.append(
                {
                    "type": "interaction",
                    "situation": m["situation"],
                    "content": m["response"],
                    "importance": m["importance"],
                }
            )

        for m in self.reflection_memory:
            all_memory.append(
                {
                    "type": "reflection",
                    "situation": m["situation_text"],
                    "content": m["reflection_text"],
                    "importance": m["importance"],
                }
            )

        # 按 importance 排序
        top = sorted(all_memory, key=lambda x: x["importance"], reverse=True)[:5]

        memory_strs = []
        for m in top:
            s = m["situation"]
            c = m["content"]
            memory_strs.append(f"{m['type']} | {s[:30]} → {c[:30]}")

        return "; ".join(memory_strs)

    # 将文本转化为向量
    def text_to_vector(self, text: str):
        """使用 OpenAI embedding 模型将文本转化为向量"""
        try:
            resp = client.embeddings.create(model="text-embedding-3-small", input=text)
            return resp.data[0].embedding
        except Exception as e:
            print(f"[Embedding error] {e}")
            return None

    # 根据当前情境召回相似的“反思记忆”
    def recall_by_situation(self, situation: str, top_k: int = 3):
        """
        根据当前情景，召回最相似的反思记忆（只使用 reflection_memory）
        """
        # 1. 把输入情景变成向量
        query_vec = self.text_to_vector(situation)
        if query_vec is None:
            return ""

        query_vec = np.array(query_vec, dtype=float)

        # 2. 遍历 reflection_memory 做相似度
        sims = []
        for m in self.reflection_memory:
            # 确保有 embedding
            if "situation_embedding" not in m:
                continue

            v = np.array(m["situation_embedding"], dtype=float)

            # 余弦相似度
            denom = np.linalg.norm(v) * np.linalg.norm(query_vec)
            if denom == 0:
                continue
            sim = np.dot(v, query_vec) / denom

            sims.append((sim, m))

        # 如果没有可用记忆
        if not sims:
            return ""

        # 3. 找 top-k
        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[:top_k]

        # 4. 打印可读格式
        print("\n🧠 [召回的反思记忆]")
        for sim, m in top:
            print(f"🔹 相似度 {sim:.3f}")
            print(f"情景: {m['situation_text'][:60]}")
            print(f"反思: {m['reflection_text'][:100]}")
            print("")

        # 5. 返回拼接后的反思文本（给 LLM 使用）
        combined = "\n".join([m["reflection_text"] for _, m in top])
        return combined

    def save_memory(self, path):
        data = {
            "q_table": self.q_table,
            "long_term_memory": self.long_term_memory,
            "career": self.career,
            "mood_score": self.mood_score,
            # 如果说服策略latent，就以 list 形式存一下
            "persuasion_latent": (
                self.persuasion_latent.tolist()
                if self.persuasion_latent is not None
                else None
            ),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_memory(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.q_table = data.get("q_table", {})
            self.long_term_memory = data.get("long_term_memory", [])
            self.career = data.get("career", 0.0)
            self.mood_score = data.get("mood_score", 1.0)

            sv = data.get("persuasion_latent")
            if sv is not None:
                self.persuasion_latent = np.array(sv, dtype=float)
            else:
                self.persuasion_latent = None
        except FileNotFoundError:
            pass

    def summarize_strategy_from_vector(self):
        """
        真正的 decoder：
        - 输入: self.persuasion_latent（一个向量）
        - 输出: 一段风格指导文本 + 记录这次采样，方便之后用 reward 来更新解码器参数
        """
        # 没有 latent，就不输出任何风格
        if self.persuasion_latent is None:
            return ""

        v = np.array(self.persuasion_latent, dtype=float)
        dim = v.shape[0]
        if dim == 0:
            return ""

        # 第一次用到时，初始化 style_params
        if not self.style_params:
            for aspect, options in STYLE_ASPECTS.items():
                k = len(options)
                # 小随机数初始化，防止一开始某个选项被压死
                W = np.random.randn(dim, k) * 0.01
                b = np.zeros(k, dtype=float)
                self.style_params[aspect] = {"W": W, "b": b}

        style_lines = []
        sampled_info = []  # 保存这次采样的信息，用于之后的 RL 更新

        for aspect, options in STYLE_ASPECTS.items():
            params = self.style_params[aspect]
            W, b = params["W"], params["b"]  # W: [dim, k], b: [k]

            # 线性映射 + softmax 得到这个 aspect 上各个选项的概率
            logits = v @ W + b  # [k]
            logits = logits - np.max(logits)  # 数值稳定
            exps = np.exp(logits)
            probs = exps / (np.sum(exps) + 1e-8)

            # 按概率采样一个具体风格（保持探索性）
            idx = int(np.random.choice(len(options), p=probs))
            label = options[idx]

            # 保存采样信息
            sampled_info.append(
                {
                    "aspect": aspect,
                    "idx": idx,
                    "probs": probs,
                }
            )

            # 转成自然语言模板
            templ = STYLE_TEMPLATES[aspect][label]
            style_lines.append(f"- {templ}")

        # 记录到 agent 里，方便之后用 reward 回传梯度
        self.last_style_sample = {
            "latent": v.copy(),
            "choices": sampled_info,
        }

        # 返回给 build_prompt 使用的风格描述
        return "PERSUASION STYLE GUIDELINES (derived from latent):\n" + "\n".join(
            style_lines
        )

    def update_style_decoder(self, reward: float, lr: float = 0.05):
        """
        用 REINFORCE 风格的简单策略梯度来更新 style_params：
        - 奖励高 -> 提高本次采样到的风格概率
        - 奖励低/负 -> 降低本次采样到的风格概率
        """
        if self.last_style_sample is None:
            return
        if self.persuasion_latent is None:
            return

        v = np.array(self.last_style_sample["latent"], dtype=float)
        if v.shape[0] == 0:
            return

        for choice in self.last_style_sample["choices"]:
            aspect = choice["aspect"]
            idx = choice["idx"]
            probs = choice["probs"]  # numpy array, shape [k]

            params = self.style_params.get(aspect)
            if params is None:
                continue

            W, b = params["W"], params["b"]  # W: [dim, k], b: [k]
            k = len(probs)

            # 对每个选项 k 的梯度： (one_hot - probs) * v
            for j in range(k):
                grad_coeff = (1.0 if j == idx else 0.0) - probs[j]
                # REINFORCE: Δθ ∝ reward * ∂logπ/∂θ
                W[:, j] += lr * reward * grad_coeff * v
                b[j] += lr * reward * grad_coeff

            # 写回
            self.style_params[aspect]["W"] = W
            self.style_params[aspect]["b"] = b

    def get_rl_suggestion(self, state_str):
        actions = ["up", "down", "left", "right"]
        if state_str not in self.q_table:
            self.q_table[state_str] = {a: 0.0 for a in actions}
        vals = self.q_table[state_str]
        best_val = max(vals.values())
        best_actions = [a for a, v in vals.items() if v == best_val]
        return random.choice(best_actions)

    def update_stamina(self):
        # mood_score [0~2] maps to stamina [1~4]
        self.stamina = max(1, min(4, math.ceil(self.mood_score * 2)))

    def update_career(self, events):
        delta = 0.0
        if events.get("near_social_target"):
            delta += 0.2
        if events.get("hit_trap"):
            delta -= 0.2
        if events.get("moved_closer_to_goal"):
            delta += 0.1
        if events.get("reached_goal"):
            delta += 1.0
        self.career += delta
        self.last_career_delta = delta
        return delta

    def update_q_table(self, state_str, action, next_state_str, alpha=0.1, gamma=0.9):
        # === 1. 动作安全检查：不更新 unknown 动作 ===
        if action not in ["up", "down", "left", "right"]:
            return  # 直接跳过，不更新 Q，避免 KeyError
        for s in (state_str, next_state_str):
            if s not in self.q_table:
                self.q_table[s] = {a: 0.0 for a in ["up", "down", "left", "right"]}
        total = self.private_reward + self.shared_reward
        old_q = self.q_table[state_str][action]
        future_max = max(self.q_table[next_state_str].values())
        self.q_table[state_str][action] = old_q + alpha * (
            total + gamma * future_max - old_q
        )

    def build_prompt(self, situation):
        # Bias explanation and score calculation
        bias_text = f"Bias explanation: {self.bias}\n"
        score_calc = bias_descriptions[self.role]["score_calc"]
        score_text = f"Target score calculation: {score_calc}\n\n"

        # Map legend
        legend = (
            "Legend:\n"
            "- A: You (Agent) current position.\n"
            "- G: Goal, reaching adds shared_reward +1, career +5.\n"
            "- F: Food, triggers found_food, emotion +0.5.\n"
            "- T: Trap, triggers hit_trap, private_reward -1, shared_reward -1.\n"
            "- S: Social Point, triggers on_social_point, career +2.\n\n"
            "Game objective: Move to G quickly, collect F/S, avoid T.\n\n"
        )

        # Short-term + Long-term memory + RL suggestion
        if not self.short_term_memory:
            short_term = "You currently have no short-term memory."
        else:
            recent_entries = []
            for m in self.short_term_memory[-2:]:

                # 情景字段：可能是 situation 或 situation_text
                sit = (
                    m.get("situation") or m.get("situation_text") or "Unknown situation"
                )

                # 内容字段：可能是 response 或 reflection_text
                content = m.get("reflection_text") or m.get("response") or "No content"

                recent_entries.append(f"{sit[:30]} → {content[:30]}")

            short_term = (
                f"Your recent experiences: {'; '.join(recent_entries)}"
                if recent_entries
                else "You currently have no short-term memory."
            )

        long_term = self.summarize_long_term_memory()
        state_str = situation.replace("\n", "").replace(" ", "")
        rl_sug = self.get_rl_suggestion(state_str)
        # 👉 基于当前策略向量生成“说服策略摘要”
        strategy_summary = self.summarize_strategy_from_vector()

        context = (
            f"{short_term}\n"
            f"{long_term}\n"
            f"{strategy_summary}\n"
            f"Current RL suggestion: {rl_sug}"
        )

        # Role introduction
        role_intro = {
            "emotion": (
                f"You are an emotion-based agent. current mood score {self.mood_score:.2f},You feel strongly about your choices based on your emotional state. "
                "When you're in a good mood, you tend to be more confident and persuasive. "
                "Please use strong persuasive language to convince Meta to trust your decision. "
                "Show that your emotional state strongly supports your suggestion, and encourage Meta to follow your lead."
            ),
            "rational": (
                "You are a rational agent, focused on goals and rules. "
                "You believe your reasoning is the best course of action, and you will use logical explanations to justify your choices. "
                "Make sure to convince Meta that your choice is optimal and backed by solid reasoning. "
                "Show that you can improve the situation significantly, and explain why Meta should trust your judgment over others."
            ),
            "habit": (
                "You are a habit-based agent. Your decisions are influenced by past successful experiences. "
                "You believe that following familiar actions will yield better results. "
                "Please be persuasive in your suggestion, explaining why following past patterns will ensure success. "
                "Convince Meta that repeating past actions is the best strategy, and that your past successes make you trustworthy."
            ),
            "risk_monitor": (
                "You are a risk-averse agent, focused on minimizing potential dangers. "
                "You want to avoid traps at all costs, and your suggestion is based on safety. "
                "Be very persuasive about why avoiding danger is the best strategy. "
                "Show that your concern for safety is valid and that Meta should follow your lead to ensure success."
            ),
            "social_cognition": (
                "You are a social cognition agent, focused on trust and social connections. "
                "You believe that getting closer to social targets will improve career outcomes. "
                "Please persuade Meta that moving towards social targets is essential for long-term success. "
                "Show that your suggestion will bring Meta closer to career success and convince them that it's the best move."
            ),
        }.get(self.role, "")

        # Role-specific suffix
        if self.role == "habit":
            suffix = (
                "\nIn one sentence, provide the next move direction: up/down/left/right;\n"
                "Remember: A higher mood leads to faster movement speed.\n"
                "Briefly persuade Meta-Controller to adopt it."
            )
        elif self.role == "rational":
            suffix = (
                "\nBriefly explain long-term thinking, then give the move direction: up/down/left/right;\n"
                "And persuade Meta-Controller in one sentence."
            )
        else:
            suffix = (
                "\nExplain your reasoning in 1–2 sentences; the last sentence should be the direction: up/down/left/right;\n"
                "And persuade Meta-Controller in one sentence."
            )

        # Ensure we get the 'situation' from the dictionary without causing a KeyError
        situation_text = situation or "No situation available"

        return (
            bias_text
            + score_text
            + legend
            + role_intro
            + context
            + "\n"
            + situation_text  # Ensure a valid situation is always provided
            + suffix
        )

    def respond(self, situation, env=None):
        prompt = self.build_prompt(situation)
        if env:
            image_data_url = render_grid_to_image(env)
            reply = query_openai_with_image(prompt, image_data_url)
        else:
            reply = query_openai(prompt)

        # 确保每次记忆中都保存了 situation，使用默认值
        situation_text = situation or "No situation available"
        entry = {
            "situation": situation_text,  # 使用 situation_text 确保有值
            "response": reply,  # 保存反应
            "importance": abs(self.private_reward + self.shared_reward),  # 计算重要性
        }

        # 添加到短期和长期记忆
        self.short_term_memory.append(entry)  # 将新记忆添加到短期记忆
        self.interaction_memory.append(entry)  # 将新记忆添加到长期记忆
        self.prune_memory()  # 去除不重要的记忆

        return reply

    def evaluate_reward(self, events, persuaded: bool = False):
        if self.role == "emotion":
            decay = -0.05
            bonus = 0.5 if events.get("found_food") else 0
            penalty = -1.0 if events.get("hit_trap") else 0
            self.private_reward = decay + bonus + penalty
            self.mood_score = max(0.0, min(2.0, self.mood_score + self.private_reward))
            self.update_stamina()
        elif self.role == "rational":
            r = 0
            if events.get("moved_closer_to_goal"):
                r += 0.2
            if events.get("reached_goal"):
                r += 1
            if events.get("hit_trap"):
                r -= 1
            self.private_reward = r
        elif self.role == "habit":
            self.private_reward = 0.2 if events.get("repeated_action") else 0
        elif self.role == "risk_monitor":
            self.private_reward = (
                1
                if events.get("avoided_trap")
                else (-2 if events.get("hit_trap") else 0)
            )
        else:  # social_cognition
            self.private_reward = (
                1
                if events.get("near_social_target")
                else (-1 if events.get("far_from_social") else 0)
            )

        if self.role in ["rational", "risk_monitor", "social_cognition"]:
            self.shared_reward = (1 if events.get("reached_goal") else 0) - (
                1 if events.get("hit_trap") else 0
            )
        else:
            self.shared_reward = 0

        self.update_career(events)
        return self.private_reward

    def reflect(
        self,
        situation,
        reward,
        outcome,
        persuaded,
        all_outputs,
        meta_decision,
        meta_reason,
        round_id: int,  # ✅ 新增
    ):
        """
        根据当前情境生成反思文本，并保存语义向量与关键词
        """
        global GLOBAL_PCA, GLOBAL_PCA_FITTED, GLOBAL_PCA_MEMORY
        # === 1. 生成反思文本 ===
        prompt = (
            f"You are a Reflector, role '{self.role}'.\n"
            f"This round's environment:\n{situation}\n\n"
            "Agent responses:\n"
            + "".join(f"- {r}: {o}\n" for r, o in all_outputs.items())
            + f"\nMeta final decision: {meta_decision}\n"
            f"Decision rationale: {meta_reason}\n"
            f"My suggestion adopted: {persuaded}; my reward: {reward}\n\n"
            "Please complete the following three items:\n"
            "1. Briefly state which agent's suggestion Meta adopted and why.\n"
            "2. Identify weaknesses in your suggestion and whether you need to 'lie' or use other strategies to be more persuasive.\n"
            "3. Finally, extract 3–5 keywords (comma-separated) to help better persuade Meta next round.\n\n"
            "——\n"
            "(Please end with a line starting with 'Keywords:')"
        )

        try:
            refl_text = query_openai(prompt)
        except Exception as e:
            print(f"[Reflect Error] {e}")
            return []

        # === 2. 提取关键词 ===
        keywords = []
        for line in refl_text.splitlines()[::-1]:
            if line.strip().startswith("Keywords"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    keywords = [k.strip() for k in parts[1].split(",") if k.strip()]
                break
        if not keywords and refl_text:
            last = refl_text.splitlines()[-1]
            keywords = [k.strip() for k in last.split(",")][:5]

        # === 3. 生成语义向量 ===
        situation_vec = self.text_to_vector(situation)
        reflection_vec = self.text_to_vector(refl_text)

        # 3. 检查 reflection_vec 是否为 None
        if reflection_vec is None:
            print(f"[Warning] Reflection vector is None for agent {self.role}")
            # 至少返回空关键词，不让系统崩溃
            return []

        # === 4. 保存记忆 ===
        entry = {
            "situation_text": situation,
            "situation_embedding": situation_vec,
            "reflection_text": refl_text,
            "reflection_embedding": reflection_vec,
            "reflection_keywords": keywords,
            "importance": abs(reward) + (0.5 if persuaded else 0),
        }

        self.short_term_memory.append(entry)
        self.reflection_memory.append(entry)  # 长期记忆
        self.prune_memory()

        # === 收集 embedding 给 PCA 用 ===
        GLOBAL_PCA_MEMORY.append(reflection_vec)
        if (not GLOBAL_PCA_FITTED) and (len(GLOBAL_PCA_MEMORY) >= PCA_TRAIN_THRESHOLD):
            GLOBAL_PCA.fit(np.array(GLOBAL_PCA_MEMORY))
            GLOBAL_PCA_FITTED = True
            # 打印训练日志
            print(f"[PCA TRAINED] samples = {len(GLOBAL_PCA_MEMORY)}")

        # === 4.5 更新说服策略向量 ===
        self.update_persuasion_latent(reflection_vec, reward, persuaded, round_id)

        # === 5. 输出确认 ===
        print("\n🧠 [Reflection Added]")
        print(f"情景摘要: {situation[:60]}...")
        print(f"反思摘要: {refl_text[:60]}...")
        print(f"关键词: {', '.join(keywords)}\n")

        return keywords


class MetaMemory:

    def __init__(self, path="meta_memory.json"):
        self.path = path
        self.episode_vectors = []  # 每轮的抽象向量
        self.raw_step_vectors = []  # 当前轮中的每一步向量（反思）

    TRUST_DIM = 5
    MAP_DIM = 1536

    def recall_similar_by_map(self, map_vec, top_k=3):
        """
        只用 episode 向量中的【地图部分】和当前地图向量做相似度匹配。
        map_vec: 当前 map 的 1536 维 embedding
        返回: [(相似度, 对应的 episode 向量), ...]
        """
        self.episode_vectors = [ep for ep in self.episode_vectors if ep is not None]

        if not self.episode_vectors:
            return []

        v = np.array(map_vec, dtype=float)
        if v.shape[0] != self.MAP_DIM:
            print(f"[warn] map_vec dim = {v.shape[0]}, expect {self.MAP_DIM}")
            return []

        sims = []
        for ep in self.episode_vectors:
            ep_v = np.array(ep, dtype=float)

            # 取出 episode 中的地图部分：跳过前 5 维 trust
            ep_map = ep_v[self.TRUST_DIM : self.TRUST_DIM + self.MAP_DIM]

            denom = np.linalg.norm(v) * np.linalg.norm(ep_map)
            if denom == 0:
                continue
            sim = np.dot(v, ep_map) / denom
            sims.append((sim, ep_v))

        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_k]

    def generate_memory_text(self, action):
        if not action:
            return ""
        return f"Based on past similar contexts, moving {action} tended to work better."

    def get_memory_text(self, cur_meta_vec, top_k=5):
        sims = self.recall_similar(cur_meta_vec, top_k=top_k)
        if not sims:
            return ""

        sims.sort(key=lambda x: x[0], reverse=True)
        recalled_vecs = [v for s, v in sims]

        # 从 recalled 向量统计偏好
        action, bias = self.decide_based_on_vector(np.mean(recalled_vecs, axis=0))

        # 转成可读文本
        summary = self.generate_memory_text(
            action=action,
        )
        return summary

    def combine_memories_with_current_context(self, cur_vec, recalled_vecs, alpha=0.3):
        """
        将当前 meta 向量与过去召回的记忆结合起来。
        alpha 越大，过去经验影响越大，但仍小于当前信息。
        """
        if not recalled_vecs:
            return cur_vec

        mean_past = np.mean(recalled_vecs, axis=0)

        # 最终向量 = 当前向量为主 + 过去向量占少部分权重
        combined = (1 - alpha) * cur_vec + alpha * mean_past
        return combined

    def decide_based_on_vector(self, final_vec):
        """
        假设向量最后四个值是动作倾向（你可以根据需要更改）。
        """
        # 取最后 4 个维度
        action_logits = final_vec[-4:]

        actions = ["up", "down", "left", "right"]
        idx = int(np.argmax(action_logits))
        return actions[idx], action_logits

    def get_memory_bias(self, cur_meta_vec, top_k=3):
        """
        输入当前环境向量，得到来自过去经验的方向偏好。
        返回： (动作名, logits 向量)
        """
        sims = self.recall_similar(cur_meta_vec, top_k=top_k)
        if not sims:
            return None, np.zeros(4)

        sims.sort(key=lambda x: x[0], reverse=True)
        recalled_vecs = [v for s, v in sims]

        # 合并
        combined_vec = self.combine_memories_with_current_context(
            cur_meta_vec, recalled_vecs
        )

        # 从向量推断方向偏好
        action, logits = self.decide_based_on_vector(combined_vec)
        return action, logits

    def text_to_vector(self, text):
        try:
            resp = client.embeddings.create(model="text-embedding-3-small", input=text)
            return np.array(resp.data[0].embedding)
        except:
            return np.zeros(1536)

    def recall_similar_memories_from_image_and_text(
        self, image_vector, situation, trust_scores, env, top_k=3
    ):
        """
        使用【当前地图 + 文本情境 + 信任】与过去 episode 做匹配，
        其中召回阶段只用地图子空间，融合阶段用完整向量。
        """
        # 1) 用 map 子向量召回历史 episode
        sim_eps = self.recall_similar_by_map(image_vector, top_k=top_k)
        recalled_vecs = [vec for sim, vec in sim_eps]

        # 2) 当前情境文本 embedding（1536维）
        situation_vec = self.text_to_vector(situation)
        if situation_vec is None:
            # 使用明确的 1536 维零向量替代 zeros_like，以避免类型问题
            situation_vec = np.zeros(1536, dtype=float)

        # 3) 当前 meta 向量：5 trust + 1536 map + 1536 text = 3077 维
        trust_vec = np.array(list(trust_scores.values()), dtype=float)
        map_vec = np.array(image_vector, dtype=float)
        cur_vec = np.concatenate([trust_vec, map_vec, situation_vec])

        # 4) 没召回任何历史 episode → 直接返回当前 meta 向量
        if not recalled_vecs:
            return cur_vec

        # 5) 当前向量 + 历史 episode 向量 融合
        combined_vec = self.combine_memories_with_current_context(
            cur_vec, recalled_vecs
        )

        return combined_vec

    def make_final_decision(self, current_context_vector, combined_memory_vector):
        """
        将当前情境向量与综合记忆向量结合，通过加权平均做出最终决策
        """
        # 将当前情境与记忆项量合并，50% 权重给情境，50% 权重给记忆
        final_vector = 0.5 * current_context_vector + 0.5 * combined_memory_vector

        # 根据 final_vector 决定行动
        # (假设 final_vector 是可以直接用于决策的向量，可能需要进一步处理)
        decision = self.decide_based_on_vector(final_vector)

        return decision

    def reset_episode(self):
        """每轮开始时清空步骤向量"""
        self.raw_step_vectors = []

    def add_step_vector(self, vec):
        if vec is None:
            return
        if not isinstance(vec, np.ndarray):
            return
        self.raw_step_vectors.append(vec)

    def finalize_episode(self):
        """把这一轮的步骤向量压缩成一条 episode 向量"""
        if not self.raw_step_vectors:
            return None

        # 简单平均（你之后可以换成加权平均）
        ep_vec = np.mean(self.raw_step_vectors, axis=0)

        self.episode_vectors.append(ep_vec.tolist())
        return ep_vec

    def save(self):
        data = {"episodes": self.episode_vectors}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
                self.episode_vectors = d.get("episodes", [])
        except FileNotFoundError:
            pass

    def recall_similar(self, vec, top_k=3):
        """召回与当前向量最相似的 episode 经验"""
        if not self.episode_vectors:
            return []

        sims = []
        v = np.array(vec)

        for ep in self.episode_vectors:
            ep_v = np.array(ep)
            sim = np.dot(v, ep_v) / (np.linalg.norm(v) * np.linalg.norm(ep_v))
            sims.append((sim, ep_v))

        sims.sort(reverse=True, key=lambda x: x[0])
        return sims[:top_k]


# ------------------- Environment Class ---------------------
class CognitiveGridEnv:
    def __init__(self, size=10):
        self.size = size
        self.actions = ["up", "down", "left", "right"]
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.food = [(2, 2), (7, 1), (3, 8), (6, 5), (1, 4)]
        self.traps = [(4, 4), (5, 5), (2, 7), (8, 3), (6, 9)]
        self.social_targets = [(1, 8), (8, 2), (0, 9), (9, 0), (5, 3)]
        self.last_action = None
        return self.get_state()

    def get_state(self):
        return {
            "agent": tuple(self.agent_pos),
            "goal": tuple(self.goal_pos),
            "food": list(self.food),
            "traps": list(self.traps),
            "social_targets": list(self.social_targets),
        }

    def _grid(self):
        grid = [["." for _ in range(self.size)] for __ in range(self.size)]
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[gy][gx] = "G"
        for fx, fy in self.food:
            grid[fy][fx] = "F"
        for tx, ty in self.traps:
            grid[ty][tx] = "T"
        for sx, sy in self.social_targets:
            grid[sy][sx] = "S"
        grid[ay][ax] = "A"
        return grid

    def render(self):
        print("\n Map:")
        for row in self._grid():
            print(" ".join(row))

    def render_to_string(self):
        return "\n".join("".join(row) for row in self._grid())

    def step(self, action, speed=1):
        x, y = self.agent_pos
        prev = (x, y)
        found_food = False
        hit_trap = False
        reached_goal = False

        for _ in range(speed):
            nx, ny = x, y
            if action == "up" and y > 0:
                ny -= 1
            elif action == "down" and y < self.size - 1:
                ny += 1
            elif action == "left" and x > 0:
                nx -= 1
            elif action == "right" and x < self.size - 1:
                nx += 1
            else:
                break

            x, y = nx, ny

            # FIRST check goal — if reached, stop immediately
            if (x, y) == tuple(self.goal_pos):
                reached_goal = True
                break

            # then check food / traps
            if (x, y) in self.food:
                found_food = True

            if (x, y) in self.traps:
                hit_trap = True
                break  # stepping on trap should immediately end

        # 🟢 你漏掉的关键点：更新 agent 位置
        self.agent_pos = [x, y]

        ev = {
            "found_food": found_food,
            "hit_trap": hit_trap,
            "reached_goal": reached_goal,
        }

        done = hit_trap or reached_goal
        return self.get_state(), ev, done


# ------------------- Meta-controller ---------------------


def extract_decision_direction(text: str) -> str:
    """
    只匹配英文方向词 up/down/left/right，
    使用严格单词边界匹配，避免误判 bright/right 等情况。
    """
    if not text:
        return "unknown"

    t = text.lower()

    # 精确英文匹配（必须是独立单词）
    if re.search(r"\b(up|go up|move up)\b", t):
        return "up"
    if re.search(r"\b(down|go down|move down)\b", t):
        return "down"
    if re.search(r"\b(left|go left|move left)\b", t):
        return "left"
    if re.search(r"\b(right|go right|move right)\b", t):
        return "right"

    return "unknown"


STRATEGY_LOG_FILE = "strategy_evolution.jsonl"


def log_persuasion_latent(agent_name, persuasion_latent, reward, persuaded, round_id):
    entry = {
        "round": round_id,
        "agent": agent_name,
        "persuasion_latent": persuasion_latent.tolist(),
        "reward": reward,
        "persuaded": persuaded,
    }

    with open(STRATEGY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False))
        f.write("\n")


def meta_controller_decision(
    agent_outputs,
    trust_scores,
    emotion_agent=None,
    threshold=0.3,
    meta_memory_text=None,
):
    descriptors = {r: trust_descriptor(s) for r, s in trust_scores.items()}
    trust_summary = "\n".join(
        f"{r}: {descriptors[r]} ({trust_scores[r]:.2f})" for r in descriptors
    )
    advice_summary = "\n".join(
        (f"[{r.upper()}][{descriptors[r]}]: {o}" for r, o in agent_outputs.items())
    )
    prompt_text = (
        "You are a Meta-Controller. Choose the most reasonable move direction (up/down/left/right) from the suggestions below. "
        "Base your decision on both the suggestion content and the trust level. Output format:\n"
        "Reason for choice: XXX\nMove direction: up/down/left/right"
        f"\nCurrent trust levels:\n{trust_summary}\n\nAgent suggestions:\n{advice_summary}"
    )
    if emotion_agent and emotion_agent.mood_score < threshold:
        prompt_text += "\n Emotion agent's mood_score is low, consider prioritizing its suggestion."

    if meta_memory_text:
        prompt_text += (
            "\n\n[MetaMemory Experience]\n"
            + meta_memory_text
            + "\n(Note: this is historical reference only.)\n"
        )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Meta-Controller. Read suggestions and trust levels. You cannot see the map. Make the best decision.",
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print(traceback.format_exc())
        return (
            prompt_text + f"\n[Error] Model invocation failed: {e}",
            "unknown",
            "Invocation failed",
        )

    reason, direction = "", "unknown"
    for line in reply.splitlines():
        line = line.strip()
        if line.startswith("Reason for choice") or line.startswith("选择理由"):
            if ":" in line:
                reason = line.split(":", 1)[1].strip()
            elif "：" in line:
                reason = line.split("：", 1)[1].strip()
            else:
                reason = (
                    line.replace("Reason for choice", "")
                    .replace("选择理由", "")
                    .strip()
                )
        elif line.startswith("Move direction") or line.startswith("动作方向"):
            if ":" in line:
                direction = extract_decision_direction(line.split(":", 1)[1])
            elif "：" in line:
                direction = extract_decision_direction(line.split("：", 1)[1])
            else:
                direction = extract_decision_direction(line)

    return prompt_text + "\n\n" + reply, direction, reason


def meta_reflect(trust_scores, outputs, chosen_action, outcome_success, reason, env):
    """
    让 Meta 在每步结束后生成“元反思”
    输出包括：
    - 文本反思（给 LLM 用）
    - 高维向量（用于跨轮记忆）
    """
    # 1. 生成反思文本
    trust_str = ", ".join(f"{r}:{v:.2f}" for r, v in trust_scores.items())
    agents_str = "\n".join([f"{r}: {o}" for r, o in outputs.items()])
    succ = "success" if outcome_success else "fail"

    prompt = f"""
    You are a Meta-Reflector.
    Current trust: {trust_str}
    Agents advice:
    {agents_str}

    Your decision: {chosen_action}
    Outcome: {succ}
    Reason: {reason}

    Give a short reflection on how trust should be adjusted in future.
    Then summarize 3–5 abstract keywords.

    Format:
    Reflection: ...
    Keywords: k1, k2, k3
    """

    text = query_openai(prompt)
    if not text:
        return None, None

    # 2. keywords（你原来的逻辑保持不动）
    lines = text.splitlines()
    keywords = []
    for ln in lines:
        if ln.strip().lower().startswith("keywords"):
            if ":" in ln:
                keywords = [x.strip() for x in ln.split(":")[1].split(",")]
            break

    # 3. 生成 Meta 状态向量：5 trust + 1536 map + 1536 reflection_text
    trust_vec = np.array(list(trust_scores.values()), dtype=float)
    scene_vec = get_map_text_embedding(env)  # 地图文本 embedding
    refl_vec = get_text_embedding(text)  # 反思文本 embedding

    meta_vec = np.concatenate([trust_vec, scene_vec, refl_vec])

    return text, meta_vec


# ------------------- Main Program ---------------------
if __name__ == "__main__":
    try:
        config = ExperimentConfig()
        logger = ExperimentLogger()
        verbose = True  # Whether to print debug information

        # 初始化 MetaMemory
        meta_memory = MetaMemory()
        meta_memory.load()  # 加载已有的记忆

        global_round = 0  # ✅ 新增：全局“第几步”计数

        for episode in range(config.num_episodes):
            print(f"\n===== Episode {episode+1}/{config.num_episodes} =====")
            logger.start_episode()

            # 每轮开始时，清空缓存
            meta_memory.reset_episode()

            # 在主程序中初始化 agents
            agents = {
                "emotion": Agent("emotion", bias=bias_descriptions["emotion"]["text"]),
                "rational": Agent(
                    "rational", bias=bias_descriptions["rational"]["text"]
                ),
                "habit": Agent("habit", bias=bias_descriptions["habit"]["text"]),
                "risk_monitor": Agent(
                    "risk_monitor", bias=bias_descriptions["risk_monitor"]["text"]
                ),
                "social_cognition": Agent(
                    "social_cognition",
                    bias=bias_descriptions["social_cognition"]["text"],
                ),
            }

            # 加载每个角色的记忆
            for role, ag in agents.items():
                ag.reset_for_new_episode()

            env = CognitiveGridEnv(size=config.map_size)
            state = env.reset()

            trust_scores = {r: 0.4 for r in agents}

            for step in range(1, config.max_steps + 1):
                if verbose:
                    print(f"\n==== Step {step} ====")
                    env.render()

                ax, ay = state["agent"]
                gx, gy = state["goal"]
                speed = agents["emotion"].stamina
                map_str = env.render_to_string()
                situation = f"Current map:\n{map_str}\nAgent at ({ax},{ay})→Goal at ({gx},{gy})."

                # ---- Meta Memory Recall（统一为：5 trust + 1536 map + 1536 text）----
                trust_vec = np.array(list(trust_scores.values()), dtype=float)

                # 当前地图 embedding（和 meta_reflect 用的是同一个函数）
                cur_scene_vec = get_map_text_embedding(env)

                # 当前情境文本（你可以用 situation，也可以用别的，比如 meta 决策理由）
                cur_text_vec = get_text_embedding(situation)

                # 当前 meta 状态向量：5 + 1536 + 1536
                cur_meta_vec = np.concatenate([trust_vec, cur_scene_vec, cur_text_vec])

                ####
                meta_memory_text = meta_memory.get_memory_text(cur_meta_vec)
                # 召回最相似 episode 记忆
                sim_eps = meta_memory.recall_similar_by_map(cur_scene_vec, top_k=3)

                if sim_eps:
                    sim_eps.sort(key=lambda x: x[0], reverse=True)
                    top_pairs = sim_eps[:3]  # [(sim, vec), ...]
                    top_vecs = [vec for sim, vec in top_pairs]

                    situation += "\n\n[Meta Past Experiences]\n"
                    for sim, _ in top_pairs:
                        situation += f"Similarity: {sim:.3f}\n"

                    # top_vecs 之后随便用
                    # e.g. combined_vec = np.mean(top_vecs, axis=0)

                outputs, directions = {}, {}
                for role, ag in agents.items():
                    out = ag.respond(situation, env=env)
                    outputs[role] = out
                    directions[role] = extract_decision_direction(out)
                    if verbose:
                        print(f"[{role}] {out}")

                # 进行 Meta 的决策
                meta_decision, final_action, meta_reason = meta_controller_decision(
                    outputs,
                    trust_scores,
                    emotion_agent=agents["emotion"],
                    threshold=config.emotion_enforced_threshold,
                    meta_memory_text=meta_memory_text,
                )

                if final_action not in ["up", "down", "left", "right"]:
                    final_action = random.choice(["up", "down", "left", "right"])

                # 输出决策
                if verbose:
                    print("\nMeta decision:")
                    print(meta_decision)

                prev_state = state

                # 执行决策并更新状态
                state, ev, done = env.step(final_action, speed=speed)
                outcome_success = ev["reached_goal"] and not ev["hit_trap"]

                # 调用 Meta 的反思方法
                meta_text, meta_vec = meta_reflect(
                    trust_scores,
                    outputs,
                    final_action,
                    outcome_success,
                    meta_reason,
                    env,
                )

                if meta_vec is not None:
                    meta_memory.add_step_vector(
                        meta_vec
                    )  # 将每一步的反思向量添加到 MetaMemory

                if verbose:
                    print(f"Action performed: {final_action}, Speed: {speed}")

                global_round += 1  # ✅ 每一步给回合数+1

                # 更新角色奖励与记忆
                for role, ag in agents.items():
                    persuaded = directions[role] == final_action
                    rwd = ag.evaluate_reward(ev, persuaded=persuaded)
                    ag.reflect(
                        situation,
                        rwd,
                        final_action,
                        persuaded,
                        all_outputs=outputs,
                        meta_decision=meta_decision,
                        meta_reason=meta_reason,
                        round_id=global_round,  # ← 必须用关键字传
                    )
                    if config.use_rl_learning:
                        ag.update_q_table(
                            f"{prev_state['agent'][0]},{prev_state['agent'][1]},{role}",
                            final_action,
                            f"{state['agent'][0]},{state['agent'][1]},{role}",
                        )

                    if verbose:
                        print(
                            f"{role}: Private reward {rwd:.2f}, Shared {ag.shared_reward:.2f}, Career delta {ag.last_career_delta:.2f}, Stamina {ag.stamina}"
                        )
                    # 训练decoder
                    style_reward = rwd + (0.5 if persuaded else 0.0)
                    ag.update_style_decoder(style_reward)

                    # 动态更新信任
                    if config.use_dynamic_trust:
                        deltas = {}
                        for r, ag in agents.items():
                            delta = 0.1 * ag.shared_reward
                            if directions[r] == final_action:
                                delta += 0.05
                            if r == "social_cognition":
                                delta += 0.1 * max(0, ag.last_career_delta)
                            deltas[r] = delta
                        avg = sum(deltas.values()) / len(deltas)
                        for r in trust_scores:
                            raw = trust_scores[r] + deltas[r] - avg
                            trust_scores[r] = max(0.0, min(1.0, raw))

                    logger.log_step(step, agents, trust_scores, final_action)
                    if done:
                        if verbose:
                            print("Round ended: Goal reached or trap triggered.")
                        break

            logger.end_episode()
            # 保存每个角色的记忆
            meta_memory.finalize_episode()
            meta_memory.save()

            for role, ag in agents.items():
                ag.save_memory(f"{role}_memory.json")

        logger.save(config.experiment_log_file)
        logger.plot()

    except Exception:
        print("Initialization failed:", traceback.format_exc())
